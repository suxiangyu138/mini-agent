"""会话层：一个会话 = 一个 Agent + 一份记忆 + 一段落盘的历史。

对应设计方案 §三「企业级记忆和上下文管理」里的**会话隔离**那一节。四个维度都要隔开：

==================  ==================================================================
维度                 这里怎么隔
==================  ==================================================================
会话 ID               ``uuid4``，建了就不再复用。删掉会话 A 之后再新建，拿到的不是 A。
记忆命名空间         每个 :class:`Session` 有自己的 :class:`~agent.memory.Memory`
                     实例，窗口和摘要都长在自己身上，天然不共享。
存储层               落盘时带 ``session_id``，消息表的主键是 ``(session_id, seq)``。
生命周期             删会话走外键级联，消息和摘要跟着走；长期记忆是另一张表，
                     除非明确勾选「同时删除」，否则不牵连。
==================  ==================================================================

**「新建会话屏蔽原有会话」在这里是四条具体的动作**（少一条都不算做到）：

1. 新 ID——不复用、不递增、不猜得到；
2. 窗口清空——新 :class:`Memory` 的 ``_messages`` 是空的，旧会话聊过什么一个字都带不过来；
3. 摘要不继承——摘要是「这次会话自己压出来的」，跨会话带过去等于把上一场对话的
   结论伪装成这场对话的背景；
4. 长期记忆按开关加载——它是唯一跨会话的东西，所以必须是**显式**的：
   ``inherit_memory`` 关掉时注入空列表，而不是「默认带上」。

第 3 条和第 4 条看着矛盾，其实是同一件事的两面：跨会话该带的是**关于用户的稳定事实**，
不该带的是**上一次对话的过程**。

模块边界：这里只有「编排」，不碰 SQL（那是 :mod:`agent.store`）、不碰 HTTP
（那是 :mod:`web.server`）、不碰提示词（那是 :mod:`agent.prompt`）。
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

from agent import Agent, AgentResult, StepRecord
from agent.memory import FactExtractor, Summarizer, looks_personal
from agent.store import CATEGORIES, CATEGORY_KEYS, Fact, Store, humanize_age
from config import Config
from main import build_agent
from web.telemetry import TurnClock, metrics_payload, step_payload

logger = logging.getLogger("web.sessions")


# --------------------------------------------------------------------------- #
# 设置项
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Setting:
    """一个开关。``hint`` 是界面上那行小字——**为什么要有这个开关**，不是复述标题。"""

    key: str
    label: str
    hint: str
    default: bool = True


#: 设置页上的全部开关。默认值一律取「开」：这些都是让助手更好用的东西，
#: 想省 token 或者在意隐私的人自己关。默认关掉的选项等于没有这个功能。
SETTINGS: tuple[Setting, ...] = (
    Setting(
        "memory_enabled",
        "长期记忆",
        "把记住的事实注入每一轮对话。关掉后新写入的也不会再影响回答。",
    ),
    Setting(
        "auto_memory",
        "自动记住",
        "回答完之后自动挑出值得长期保留的信息，不需要说「记住」。",
    ),
    Setting(
        "memory_notice",
        "写入后提示",
        "记住东西时在回答下方显示一张卡片，可以当场撤销。",
    ),
    Setting(
        "inherit_memory",
        "新会话继承记忆",
        "新建会话时带上已有的长期记忆。关掉则每个新会话从零开始。",
    ),
    Setting(
        "autosave",
        "自动保存会话",
        "对话历史写入本地数据库，关掉浏览器、重启程序后还在。",
    ),
    Setting(
        "auto_summarize",
        "自动压缩历史",
        "对话变长后把早期内容压成摘要。关掉则超出窗口的对话直接丢弃。",
    ),
    Setting(
        "show_context",
        "显示上下文状态",
        "在输入框上方显示当前上下文占用了多少。",
    ),
)

DEFAULT_SETTINGS: dict[str, bool] = {item.key: item.default for item in SETTINGS}
_KNOWN_SETTINGS = frozenset(DEFAULT_SETTINGS)

#: 摘要最多压到多少字。太长的话摘要自己就成了新的上下文负担。
SUMMARY_MAX_CHARS = 300

#: 待压缩的历史超过这么多 token 才值得花一次模型调用。定得太低会「聊三句就压缩
#: 一次」，白花钱；太高则是窗口里早就看不见了才补压。
COMPRESS_THRESHOLD_TOKENS = 400

#: 标题从第一个问题里取多少字。太长会挤掉侧栏里其它会话。
TITLE_MAX_CHARS = 24

#: 清空全部会话时，界面上要求输入的那四个字。服务端也校验一遍。
CONFIRM_WORD = "确认删除"

#: 密钥的形状。有的厂商鉴权失败时会把 key 原样写进错误正文，而我们**要把错误原文
#: 发给浏览器**——这条消息会进 DOM，用户截个图就跟着走了。所以发出去之前先抹一遍。
#: 只认 ``sk-`` 和 ``Bearer`` 两种开头：写宽了会误伤正常报错里的普通单词。
_KEY_SHAPE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-.]{8,})")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """抹掉一段文字里的密钥。

    两道，因为各有各的漏法：

    1. **配置里那几个真的 key**——最准，但只认我们知道的；
    2. **形状规则**——厂商回显、第三方库自己拼出来的字符串，我们不一定认识。

    报错本身不抹：把 ``AuthenticationError: invalid api key`` 换成「处理请求时出错」，
    等于把唯一有用的排查线索扔了，而这是个跑在本机、只有自己看的服务。
    """
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return _KEY_SHAPE.sub("***", text)


class SessionError(Exception):
    """会话层能预期到的错误（找不到会话、标题为空……）。

    HTTP 层把它翻成 4xx。**内部错误不要用这个**——那些该老老实实变成 500，
    混在一起会让「用户填错了」和「程序写错了」在日志里长得一模一样。
    """


def derive_title(question: str) -> str:
    """用第一个问题当标题。

    不花模型调用去「总结标题」：那是每建一个会话就多付一次钱，换来的差别在一个
    24 字的侧栏条目上根本看不出来。取前 N 个字，界面上再做省略。
    """
    flat = " ".join((question or "").split())
    if len(flat) <= TITLE_MAX_CHARS:
        return flat or "新对话"
    return flat[:TITLE_MAX_CHARS] + "…"


# --------------------------------------------------------------------------- #
# 时间分组（侧栏上的「今天 / 昨天 / 更早」）
# --------------------------------------------------------------------------- #

#: 分组顺序 = 界面上从上到下的顺序，「置顶」永远在最前。
GROUP_LABELS: tuple[tuple[str, str], ...] = (
    ("pinned", "置顶"),
    ("today", "今天"),
    ("yesterday", "昨天"),
    ("earlier", "更早"),
)


def group_of(stamp: float, now: float, pinned: bool = False) -> str:
    """一个会话该归到哪一组。

    按**自然日**分，不是「24 小时以内算今天」。凌晨一点看昨天下午的对话，期望是
    「昨天」而不是「17 小时前」——分组是给人看的日历，不是计时器。所以这里把两个
    时间戳都折算成当地日期再比。

    日期差也刻意用 ``date`` 相减而不是除以 86400：夏令时那天只有 23 小时，
    除法的结果在切换当天会错一天。
    """
    if pinned:
        return "pinned"
    today = time.localtime(now)
    that = time.localtime(stamp)
    today_day = date(today.tm_year, today.tm_mon, today.tm_mday)
    that_day = date(that.tm_year, that.tm_mon, that.tm_mday)
    delta = (today_day - that_day).days
    if delta <= 0:
        return "today"  # 未来时间（改了系统时钟）也算今天，总比归到「更早」强
    return "yesterday" if delta == 1 else "earlier"


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #


class Session:
    """一次对话。窗口、摘要、长期事实三层的**生命周期都跟着它**。

    Agent 是懒装配的：点开侧栏里的旧会话不该触发一次模型层的初始化。真正要用才去建，
    建好之后把落盘的历史和摘要灌回去——顺序不能反，``restore`` 要求 ``Memory`` 已经存在。
    """

    def __init__(
        self,
        session_id: str,
        config: Config,
        store: Store,
        settings: dict[str, bool],
        title: str = "",
        inherit_memory: bool = True,
        turns: int = 0,
    ) -> None:
        self.id = session_id
        self.config = config
        self.store = store
        #: **不是副本**，是管理器那一份设置对象本身。管理器改设置用的是原地 update，
        #: 所以这里下一轮就能读到新值，不需要任何同步机制。存成副本反而会出现
        #: 「设置页改了、对话里没生效」这种只有重启才能解释的现象。
        self.settings = settings
        self.title = title
        self.inherit_memory = inherit_memory
        self.turns = turns

        self._lock = threading.Lock()
        self._agent: Agent | None = None
        self._running: Agent | None = None
        self._emit: Any = None
        self._clock: TurnClock | None = None

    # ------------------------------------------------------------------ #
    # 装配
    # ------------------------------------------------------------------ #

    @property
    def agent(self) -> Agent:
        """懒装配一次，顺手把落盘的历史、摘要和压缩器装上。"""
        if self._agent is None:
            agent = build_agent(self.config, on_text=self._on_text, on_step=self._on_step)
            self._load(agent)
            self._agent = agent
        return self._agent

    def _load(self, agent: Agent) -> None:
        """把库里这个会话的东西灌进它自己那份记忆。"""
        if self.settings.get("autosave", True):
            messages = self.store.load_messages(self.id)
            summary, upto = self.store.get_summary(self.id)
            if messages or summary:
                agent.memory.restore(messages, summary=summary, summary_upto=upto)
                logger.debug("会话 %s 恢复 %d 条历史", self.id[:8], len(messages))
        self.refresh_facts(agent)

    def refresh_facts(self, agent: Agent | None = None) -> None:
        """从库里重取长期记忆，注入这一轮的上下文。

        每轮开头都刷一次，是为了让「记忆管理页里改了一条」**下一轮就生效**——
        缓存住的话，用户会以为改动没保存。代价是一次本地 SQLite 查询，
        和模型调用相比可以忽略。
        """
        target = agent or self._agent
        if target is None:
            return
        target.memory.set_facts(self.injected_facts())

    def injected_facts(self) -> list[str]:
        """这个会话这一轮会看到哪几条长期记忆。

        「看到几条」这个判断只写在这里：``refresh_facts`` 拿它去注入，
        ``context()`` 拿它去显示。两处各写一份 if 迟早会漂移，而漂移的表现是
        「界面上说注入了 3 条，实际模型一条都没看到」——这种错最难查。
        """
        if not self.settings.get("memory_enabled", True) or not self.inherit_memory:
            return []
        return [fact.content for fact in self.store.list_facts(include_disabled=False)]

    def _on_text(self, chunk: str) -> None:
        if self._clock is not None:
            self._clock.mark(chunk)
        if self._emit is not None:
            self._emit("text", {"delta": chunk})

    def _on_step(self, record: StepRecord) -> None:
        # 只推「工具真的被调用」的那一步。final / max_steps 带的是最终答案，它已经走
        # text / done 两条路过去了；再当步骤推一遍，界面上就会多出一个没有名字的工具块。
        if record.kind != "tool" or self._emit is None:
            return
        self._emit("step", step_payload(record))

    def _secrets(self) -> list[str]:
        """配置里有哪几个密钥 —— 报错要走浏览器之前先按这个抹一遍。"""
        return [str(getattr(self.config, name, "") or "") for name in ("api_key", "search_api_key")]

    # ------------------------------------------------------------------ #
    # 控制
    # ------------------------------------------------------------------ #

    def cancel(self) -> bool:
        """请求中断正在跑的那一轮。线程安全，随时可调（Agent.cancel 是 Event）。"""
        running = self._running
        if running is None:
            return False
        logger.info("收到中断请求")
        running.cancel()
        return True

    def reset(self) -> None:
        """清空当前上下文。

        摘要是这次会话自己的，跟着走；长期事实跨会话，留下（见 ``Memory.clear``）。
        """
        if self._agent is not None:
            self._agent.reset()
            self._persist(self._agent)

    # ------------------------------------------------------------------ #
    # 跑一轮
    # ------------------------------------------------------------------ #

    def chat(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """跑一轮对话，把过程中的事件一个个吐出来（SSE 的 event/data 对）。

        生成器是在 HTTP 处理线程里被消费的，Agent 跑在另一个线程：这样「一边生成
        一边推」和「随时能喊停」两件事才可能同时成立——要是让处理线程自己去跑
        Agent，它在 run() 里出不来，就没人去读中断请求了。

        事件的顺序是刻意的，分界线是 ``done``：

        - ``done`` **之前**只做「必须发生而且便宜」的事——落盘。写一次本地
          SQLite 是微秒级的，放在这里换来的是「客户端随时断线都不会丢历史」；
        - ``done`` **之后**才做要花模型调用的事——抽取记忆、压缩摘要。它们的
          产出是锦上添花，而用户此刻已经在读答案了，不该为它们多等。

        为什么落盘不能挪到 ``done`` 之后：客户端一断，这个生成器会在 ``yield`` 处
        收到 ``GeneratorExit``，后面的代码**一行都不会跑**。历史丢了是没有补救的，
        少抽一条记忆没有。
        """
        with self._lock:  # 一轮到底，中途不会被第二个标签页插进来
            agent = self.agent
            self.refresh_facts(agent)
            events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
            clock = TurnClock(started=time.perf_counter())

            self._emit = lambda name, payload: events.put((name, payload))
            self._clock = clock
            self._running = agent

            box: dict[str, Any] = {}

            def work() -> None:
                try:
                    box["result"] = agent.run(message)
                except Exception as exc:  # 兜底：绝不让异常把一个 HTTP 响应悬在半空
                    logger.exception("这一轮跑挂了")
                    box["error"] = exc
                finally:
                    events.put(None)

            threading.Thread(target=work, daemon=True, name="mini-agent-turn").start()

            try:
                while True:
                    item = events.get()
                    if item is None:
                        break
                    yield item

                if "error" in box:
                    self._persist(agent)  # 跑挂了也要留下已经发生过的那部分对话
                    detail = f"{type(box['error']).__name__}: {box['error']}"
                    yield "error", {"message": redact(detail, self._secrets())}
                else:
                    result: AgentResult = box["result"]
                    self.turns += 1
                    if not self.title.strip():
                        self.title = derive_title(message)
                    self._persist(agent)
                    yield (
                        "done",
                        {
                            "answer": result.answer,
                            "metrics": metrics_payload(result, clock),
                            "session": self.brief(),
                        },
                    )
                    written = self.extract(message, result.answer, agent)
                    if written and self.settings.get("memory_notice", True):
                        yield "memory", {"facts": written}
                    compressed = self.maybe_compress(agent)
                    if compressed:
                        yield "compressed", compressed
                    yield "context", self.context(agent)
            finally:
                self._emit = None
                self._clock = None
                self._running = None
                if "result" not in box and "error" not in box:
                    # 消费者提前跑掉了（关了标签页 / 断了连接）：这一轮没人要了，
                    # 让它在下一个检查点自己收尾，别继续烧 token。
                    logger.info("客户端断开，中止这一轮")
                    agent.cancel()

    # ------------------------------------------------------------------ #
    # 一轮结束之后：落盘 / 抽取 / 压缩
    # ------------------------------------------------------------------ #

    def _persist(self, target: Agent) -> None:
        """把内存里的历史写回库。没开自动保存就什么都不做。

        参数是**必填**的，不给 ``None`` 兜底。下面 ``extract`` / ``maybe_compress``
        同理——它们都是「一轮对话的事」，而一轮对话必然有一个正在跑的 Agent。
        留一个「不传就退回去用 self._agent、还没有就静默返回」的口子，等于把
        「Agent 没建起来」写成一个安静的 no-op：调用方以为事情做了，实际什么都没发生，
        而且不报错、不留日志。这种错最难查。
        """
        if not self.settings.get("autosave", True):
            return
        try:
            memory = target.memory
            self.store.save_messages(self.id, memory.raw_messages())
            self.store.touch_session(self.id, turns=self.turns, title=self.title)
            if memory.summary:
                self.store.set_summary(self.id, memory.summary, memory.summary_upto)
            else:
                self.store.clear_summary(self.id)
        except Exception:  # 落盘失败不该让用户丢掉这一轮的回答
            logger.exception("会话落盘失败")

    def extract(self, question: str, answer: str, target: Agent) -> list[dict[str, Any]]:
        """抽出这一轮里值得长期记住的东西并写入。返回真正写进去的那几条。

        三道闸，从便宜到贵：

        1. 开关——关掉自动记忆就直接返回；
        2. :func:`~agent.memory.looks_personal` 的成本过滤——「今天几号」这种轮次
           一眼就没的可记，省下一次模型调用；
        3. 去重——提示词里已经交代过不要重复，但模型是会重复的，而且重复的代价很
           具体：同一件事攒成五条，界面上成了噪音，注入时还白占上下文。

        失败一律静默跳过。记忆是锦上添花，它出问题不该让用户连答案都拿不到。
        """
        if not self.settings.get("memory_enabled", True):
            return []
        if not self.settings.get("auto_memory", True):
            return []
        if not answer.strip() or not looks_personal(question):
            return []

        known = [fact.content for fact in self.store.list_facts(include_disabled=True)]
        seen = {_fingerprint(item) for item in known}
        try:
            found = FactExtractor(target.llm).extract(question, answer, known)
        except Exception:
            logger.exception("抽取长期记忆失败")
            return []

        written: list[dict[str, Any]] = []
        for content, category in found:
            fingerprint = _fingerprint(content)
            if fingerprint in seen:  # 同一件事不记第二遍
                continue
            seen.add(fingerprint)
            try:
                fact = self.store.add_fact(
                    uuid.uuid4().hex,
                    content,
                    category,
                    source_session=self.id,
                    source_title=self.title,
                )
            except Exception:
                logger.exception("写入长期记忆失败")
                continue
            written.append(fact.to_dict())
        if written:
            # 只记条数，不记内容：长期记忆是用户本人的事（姓名、偏好、在做的项目），
            # 写进日志就等于把它复制到了另一个没有加密、还可能被顺手贴出去的地方。
            # 要看具体记了什么，界面上有记忆页。
            logger.info("记下 %d 条长期记忆", len(written))
            self.refresh_facts(target)
        return written

    def maybe_compress(self, target: Agent) -> dict[str, Any] | None:
        """历史超了就滚一次摘要。没压返回 ``None``。

        触发点在**一轮结束之后**，不在 ``get_messages()`` 里——那里压缩一次就要多等
        一次模型调用，而用户看到的只是「界面卡住了」。
        """
        if not self.settings.get("auto_summarize", True):
            return None
        # 压缩器**每次现绑**，不在装配时挂上去：挂上去就等于把那一刻的 ``agent.llm``
        # 存了一份引用，而 llm 是会被换掉的（换模型后端、测试里塞脚本化的假模型）。
        # 存下来的那份会继续用旧模型压历史——一个只有跑了很久才看得出来、而且
        # 看起来像「新换的模型不听话」的 bug。
        target.memory.compressor = Summarizer(target.llm, max_chars=SUMMARY_MAX_CHARS)
        if not target.memory.needs_compression(COMPRESS_THRESHOLD_TOKENS):
            return None

        before = target.memory.summary
        summary = target.memory.compress()
        if summary == before:  # 压缩器没吐出东西（失败或空），当作没发生
            return None
        if self.settings.get("autosave", True):
            try:
                self.store.set_summary(self.id, summary, target.memory.summary_upto)
            except Exception:
                logger.exception("摘要落盘失败")
        return {"summary": summary, "chars": len(summary)}

    # ------------------------------------------------------------------ #
    # 读
    # ------------------------------------------------------------------ #

    def brief(self) -> dict[str, Any]:
        """侧栏里那一条需要的字段。"""
        return {
            "id": self.id,
            "title": self.title or "新对话",
            "turns": self.turns,
            "inherit_memory": self.inherit_memory,
        }

    def detail(self) -> dict[str, Any]:
        """一个会话的完整描述（切过去时返回）。

        **不碰 ``self.agent``**：只读 ``self._agent``。点开一个旧会话应该是纯读盘，
        不该顺带把模型层初始化起来。上下文状态在还没跑过的时候返回一个降级版本。
        """
        return {
            **self.brief(),
            "messages": self.transcript(),
            "context": self.context(),
        }

    def context(self, agent: Agent | None = None) -> dict[str, Any]:
        """上下文状态：给界面上那行「上下文约 3,200 tokens」和悬停详情用。

        还没跑过这一轮时（``_agent`` 是 None）返回一个**降级但不说谎**的版本：层
        分布和 token 数只有真正那份 ``Memory`` 才算得出来，就不编；而消息条数和
        记忆条数从库里数得出来，就照实给——那些数字界面上本来就要显示，
        少一个「--」比给个假的 0 好。
        """
        target = agent or self._agent
        if target is None:
            return {
                "session": self.id,
                "messages": self.store.message_count(self.id),
                "turns": self.turns,
                "layers": [],
                "tokens": 0,
                "pending_tokens": 0,
                "summary": "",
                "facts": len(self.injected_facts()),
                "max_context_chars": getattr(self.config, "max_context_chars", 0),
                "max_turns": getattr(self.config, "max_turns", 0),
                "auto_summarize": self.settings.get("auto_summarize", True),
                "memory_enabled": self.settings.get("memory_enabled", True),
                "inherit_memory": self.inherit_memory,
                "degraded": True,
            }
        memory = target.memory
        stats = memory.stats()
        return {
            "session": self.id,
            "messages": stats["messages"],
            "turns": stats["turns"],
            "layers": stats["layers"],
            "tokens": stats["tokens"],
            "window_messages": stats["window_messages"],
            "dropped_messages": stats["dropped_messages"],
            "pending_tokens": stats["pending_tokens"],
            "summary": memory.summary,
            "summary_upto": stats["summary_upto"],
            "facts": len(memory.facts),
            "max_context_chars": memory.max_context_chars,
            "window_chars": stats["window_chars"],
            "budget_chars": stats["budget_chars"],
            "max_turns": memory.max_turns,
            # 这里是「自动压缩这个功能开着吗」，不是「压缩器挂上了吗」——后者是内部
            # 实现细节（压缩器在真正要压缩的那一刻才现绑），拿它当界面上的开关状态
            # 会显示成「永远是关的」。
            "auto_summarize": self.settings.get("auto_summarize", True),
            "memory_enabled": self.settings.get("memory_enabled", True),
            "inherit_memory": self.inherit_memory,
        }

    def transcript(self) -> list[dict[str, Any]]:
        """把历史摊成界面能直接渲染的条目。

        读的是**未裁剪**的完整历史：窗口决定「发给模型什么」，这里决定「给用户看什么」。
        这两件事该分开——用户期望打开旧会话能看到自己说过的每一句话，哪怕其中一部分
        早就被摘要替代、不再进模型了。
        """
        target = self._agent
        if target is not None:
            messages = target.memory.raw_messages()
        else:
            messages = self.store.load_messages(self.id)
        items: list[dict[str, Any]] = []
        #: tool_call_id → 已经铺出去的那一步，等它的结果回来填进去
        pending: dict[str, dict[str, Any]] = {}

        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                items.append({"role": "user", "text": content})
            elif role == "assistant":
                if content.strip():
                    items.append({"role": "assistant", "text": content})
                for call in message.get("tool_calls") or []:
                    item = {
                        "role": "tool",
                        "tool": getattr(call, "name", "") or "",
                        "arguments": getattr(call, "arguments", {}) or {},
                        "result": "",
                        "is_error": False,
                    }
                    items.append(item)
                    pending[getattr(call, "id", "")] = item
            elif role == "tool":
                item = pending.get(str(message.get("tool_call_id") or ""))
                if item is not None:
                    item["result"] = content
                    item["is_error"] = bool(message.get("is_error"))
        return items


def _fingerprint(content: str) -> str:
    """去重用的指纹：忽略首尾空白和句末标点。

    「用户是后端工程师」和「用户是后端工程师。」是同一件事，两种写法模型都会给。
    """
    return " ".join((content or "").split()).rstrip("。.!！?？")


# --------------------------------------------------------------------------- #
# 会话管理
# --------------------------------------------------------------------------- #


class SessionManager:
    """所有会话 + 长期记忆 + 设置的唯一入口。

    活跃的 :class:`Session` 对象按需创建、常驻内存（一个会话一个 Agent），但**列表、
    标题、置顶这些元信息一律从库里读**——内存里那份只覆盖「已经打开过的」，
    拿它当数据源会在重启后漏掉一半。
    """

    def __init__(self, config: Config, store: Store | None = None) -> None:
        self.config = config
        self.store = store if store is not None else Store()
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._current = ""
        #: 原地更新的设置对象，被所有 Session 共享（见 ``Session.settings`` 的说明）
        self.settings: dict[str, bool] = dict(DEFAULT_SETTINGS)
        self.settings.update(
            {
                key: bool(value)
                for key, value in self.store.get_settings().items()
                if key in _KNOWN_SETTINGS
            }
        )

    # ------------------------------------------------------------------ #
    # 当前会话
    # ------------------------------------------------------------------ #

    def current(self) -> Session:
        """当前会话。没有就按「最近活动的那个」续上，一个都没有才新建。

        重启之后直接落在上次那个会话上，比每次都甩一个空白页少一次点击。
        """
        with self._lock:
            if self._current and self._current in self._sessions:
                return self._sessions[self._current]
            records = self.store.list_sessions()
            if records:
                return self._activate(records[0]["id"])
            return self._create()

    def select(self, session_id: str) -> Session:
        with self._lock:
            return self._activate(session_id)

    def _activate(self, session_id: str) -> Session:
        if session_id in self._sessions:
            self._current = session_id
            return self._sessions[session_id]
        record = self.store.get_session(session_id)
        if record is None:
            raise SessionError("会话不存在")
        session = Session(
            session_id=record["id"],
            config=self.config,
            store=self.store,
            settings=self.settings,
            title=record["title"],
            inherit_memory=bool(record["inherit_memory"]),
            turns=int(record["turns"]),
        )
        self._sessions[session_id] = session
        self._current = session_id
        return session

    def new(self, inherit: bool | None = None) -> Session:
        """新建一个会话。**旧会话不受影响**——它还在库里，随时切得回去。"""
        with self._lock:
            return self._create(inherit=inherit)

    def _create(self, title: str = "", inherit: bool | None = None) -> Session:
        session_id = uuid.uuid4().hex  # 全新 ID，绝不复用已经删掉的那个
        if inherit is None:
            inherit = self.settings.get("inherit_memory", True)
        self.store.create_session(session_id, title=title, inherit_memory=inherit)
        session = Session(
            session_id=session_id,
            config=self.config,
            store=self.store,
            settings=self.settings,
            title=title,
            inherit_memory=inherit,
        )
        self._sessions[session_id] = session
        self._current = session_id
        logger.info("新建会话 %s（继承长期记忆：%s）", session_id[:8], inherit)
        return session

    # ------------------------------------------------------------------ #
    # 列表与元信息
    # ------------------------------------------------------------------ #

    def list_sessions(self, now: float | None = None) -> dict[str, Any]:
        """侧栏要的那份列表：分好组、算好「几天前」。

        开头先 ``current()`` 一下，是因为**这份列表要跟主区指向同一个会话**：
        全新启动（或刚清空过）时库里一条都没有，而界面正对着一个空会话，
        侧栏却一条不列、``current`` 还是空串，前端只能特判这个空态——
        而这个空态下一毫秒就不成立了。让「当前会话」在这里落实，两边永远对得上。

        分组和时间用**同一个时钟**（这一个 ``stamp``），否则跨零点的那一秒里
        会出现「昨天」组下面挂着「刚刚」这种自相矛盾的条目。
        """
        current = self.current()
        stamp = time.time() if now is None else now
        items = [
            {
                "id": record["id"],
                "title": record["title"] or "新对话",
                "pinned": bool(record["pinned"]),
                "turns": int(record["turns"]),
                "updated_at": record["updated_at"],
                "age": humanize_age(record["updated_at"], stamp),
                "inherit_memory": bool(record["inherit_memory"]),
                "current": record["id"] == current.id,
            }
            for record in self.store.list_sessions()
        ]
        buckets: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in GROUP_LABELS}
        for item in items:
            buckets[group_of(item["updated_at"], stamp, item["pinned"])].append(item)
        return {
            "current": current.id,
            "count": len(items),
            "groups": [
                {"key": key, "label": label, "items": buckets[key]}
                for key, label in GROUP_LABELS
                if buckets[key]
            ],
        }

    def rename(self, session_id: str, title: str) -> None:
        title = " ".join((title or "").split())[:60]
        if not title:
            raise SessionError("标题不能为空")
        if self.store.get_session(session_id) is None:
            raise SessionError("会话不存在")
        self.store.rename_session(session_id, title)
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].title = title

    def pin(self, session_id: str, pinned: bool) -> None:
        if self.store.get_session(session_id) is None:
            raise SessionError("会话不存在")
        self.store.set_pinned(session_id, pinned)

    def delete(self, session_id: str, with_memories: bool = False) -> dict[str, Any]:
        """删一个会话。``with_memories`` 为真时，连它产生的长期记忆一起删。

        默认**不删**记忆：那是两件事。删对话是「这段聊天我不要了」，删记忆是
        「忘掉我告诉过你的事」。界面上也是两个动作，勾选框明说了才做。
        """
        if self.store.get_session(session_id) is None:
            raise SessionError("会话不存在")
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.cancel()  # 正在跑的那一轮随会话一起停掉
            self.store.delete_session(session_id)  # 消息和摘要走外键级联
            dropped = self.store.delete_facts_from_session(session_id) if with_memories else 0
            if self._current == session_id:
                self._current = ""
        logger.info("删除会话 %s（同时删除 %d 条记忆）", session_id[:8], dropped)
        return {
            "deleted": session_id,
            "memories_dropped": dropped,
            "current": self.current().id,
        }

    def clear_all(self, confirm: str) -> int:
        """清空全部会话。

        ``confirm`` 必须是「确认删除」四个字——这是界面上那个输入框的约定，在**服务端**
        再校验一遍：前端校验挡的是手滑，后端校验挡的是「有人绕开界面直接发请求」。
        """
        if (confirm or "").strip() != CONFIRM_WORD:
            raise SessionError(f"请输入「{CONFIRM_WORD}」以确认")
        with self._lock:
            for session in self._sessions.values():
                session.cancel()
            self._sessions.clear()
            self._current = ""
            count = self.store.delete_all_sessions()
        logger.warning("清空了 %d 个会话（长期记忆保留）", count)
        return count

    # ------------------------------------------------------------------ #
    # 长期记忆
    # ------------------------------------------------------------------ #

    def list_facts(self, now: float | None = None) -> dict[str, Any]:
        """记忆管理页的数据：按分类分组，附带每条的来源和「更新于」。

        被禁用的事实也返回——那个页面要让人看得见自己关掉了什么，关掉不是删掉。
        """
        stamp = time.time() if now is None else now
        facts = self.store.list_facts(include_disabled=True)
        buckets: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in CATEGORIES}
        for fact in facts:
            item = fact.to_dict()
            item["age"] = humanize_age(fact.updated_at, stamp)
            item["created_age"] = humanize_age(fact.created_at, stamp)
            item["origin"] = fact.source_title or "手动添加"
            buckets.setdefault(fact.category, buckets["other"]).append(item)
        return {
            "count": len(facts),
            "enabled": sum(1 for fact in facts if not fact.disabled),
            "categories": [
                {"key": key, "label": label, "items": buckets.get(key, [])}
                for key, label in CATEGORIES
            ],
        }

    def add_fact(self, content: str, category: str = "other") -> Fact:
        fact = self.store.add_fact(
            uuid.uuid4().hex, _clean_content(content), _clean_category(category)
        )
        self._sync_facts()
        return fact

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        disabled: bool | None = None,
    ) -> Fact:
        fact = self.store.update_fact(
            fact_id,
            content=_clean_content(content) if content is not None else None,
            category=_clean_category(category) if category is not None else None,
            disabled=disabled,
        )
        if fact is None:
            raise SessionError("这条记忆不存在")
        self._sync_facts()
        return fact

    def delete_fact(self, fact_id: str) -> None:
        if self.store.get_fact(fact_id) is None:
            raise SessionError("这条记忆不存在")
        self.store.delete_fact(fact_id)
        self._sync_facts()

    def clear_facts(self) -> int:
        count = self.store.delete_all_facts()
        self._sync_facts()
        return count

    def _sync_facts(self) -> None:
        """记忆一改，所有活着的会话立刻跟上，不用等下一轮。

        这里**不能**只刷当前会话：另一个标签页开着的是另一个会话，用户切回去看到的
        必须是改完之后的样子。
        """
        with self._lock:
            for session in self._sessions.values():
                session.refresh_facts()

    # ------------------------------------------------------------------ #
    # 设置
    # ------------------------------------------------------------------ #

    def settings_payload(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "key": item.key,
                    "label": item.label,
                    "hint": item.hint,
                    "value": self.settings[item.key],
                }
                for item in SETTINGS
            ]
        }

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """改设置。只认已知的 key、只收布尔值。

        用原地 ``update`` 而不是重新赋值：所有 Session 共享的就是这个 dict 对象，
        换掉它会让已经打开的会话继续读旧值（见 ``Session.settings`` 的说明）。
        """
        accepted = {
            key: bool(value)
            for key, value in (values or {}).items()
            if key in _KNOWN_SETTINGS and isinstance(value, bool)
        }
        if not accepted:
            raise SessionError("没有可识别的设置项")
        self.settings.update(accepted)
        self.store.set_settings(accepted)
        if "memory_enabled" in accepted:
            self._sync_facts()  # 关掉之后每个会话下一轮就不该再带记忆
        return self.settings_payload()

    def close(self) -> None:
        self.store.close()


def _clean_content(content: str | None) -> str:
    flat = " ".join((content or "").split())
    if not flat:
        raise SessionError("内容不能为空")
    if len(flat) > 200:
        raise SessionError("一条记忆最多 200 字")
    return flat


def _clean_category(category: str | None) -> str:
    value = (category or "other").strip().lower()
    return value if value in CATEGORY_KEYS else "other"
