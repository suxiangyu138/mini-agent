"""记忆层：三层上下文（最近对话 / 滚动摘要 / 长期事实）与裁剪。

对应设计方案 §三.3。核心是**裁剪时不能把 assistant 的工具调用和对应的 tool 结果拆散**
——它们必须成对保留或成对删除，否则消息格式非法，下一轮请求直接报错。

这里的做法是把历史切成「调用组」，再把调用组归入以 user 消息为界的「轮」，
裁剪永远以**整轮**为单位丢弃，从根上避免拆散配对。

三层各管一段，拼起来就是交给模型的那份上下文::

    [system 提示]                      ← 不参与裁剪，永远置顶
    [user: 长期事实 + 滚动摘要]         ← 跨会话 / 跨重启的那两层
    [最近 N 轮原始对话]                ← 窗口，信息密度最高，原样保留

**一条贯穿始终的硬规则：``get_messages()`` 绝不调用模型。** 压缩摘要要花一次模型调用，
所以它被移到了这一层之外（:meth:`Memory.compress`），由入口层在一轮对话**结束之后**触发。
读取上下文是个纯函数：不产生费用、不引入延迟、不因为模型超时而失败。这条不变式一旦破了，
「界面卡住」和「模型在压缩历史」这两件事在用户眼里就分不出来了。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any, Protocol

from .llm import ToolCall, assistant_message, system_message, tool_message, user_message
from .prompt import (
    CONTEXT_FOOTER,
    FACT_SYSTEM,
    FACT_USER,
    MEMORY_HEADER,
    SUMMARY_HEADER,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
)

logger = logging.getLogger(__name__)

#: 角色常量
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"

Message = dict[str, Any]
#: 调用组：assistant 的工具调用 + 紧随其后的 tool 结果
Group = list[Message]
#: 轮：一条 user 消息开启一轮，包含该轮里所有调用组
Turn = list[Group]


# --------------------------------------------------------------------------- #
# token 估算
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数。

    不引第三方 tokenizer 的理由和 :mod:`agent.memory` 一直以来的口径一样：
    真实分词器要么是每家厂商一份（还会随模型换代而变），要么是几百 KB 的词表。
    为了界面上一个「约 3,200 tokens」的显示值付这个代价不划算。

    经验值：中文一个字大致一个 token（常用字大多在词表里独立成词），
    英文和代码大约 4 个字符一个 token。这个数**只用来显示和算压缩阈值**，
    判断「这一轮到底花了多少」请用厂商回来的 ``input_tokens``——那个是真的。
    """
    if not text:
        return 0
    wide = 0
    for char in text:
        code = ord(char)
        # CJK 统一表意文字、CJK 标点、全角字符：都按一个字一个 token 算
        if 0x3000 <= code <= 0x9FFF or 0xFF00 <= code <= 0xFFEF:
            wide += 1
    narrow = len(text) - wide
    return wide + (narrow + 3) // 4


# --------------------------------------------------------------------------- #
# 压缩与抽取：两个要花模型调用的活
# --------------------------------------------------------------------------- #


class Compressor(Protocol):
    """摘要压缩接口。

    入参是**已有摘要**和**这次新挤出窗口的那段消息**，返回新摘要的正文。
    把旧摘要一起传进去（而不是只传新掉出来的那段）是滚动压缩的关键：
    摘要要能覆盖整段历史，而不是变成一摞互不相干的片段。
    """

    def compress(self, previous: str, messages: list[Message]) -> str: ...


def render_transcript(messages: Iterable[Message], tool_limit: int = 400) -> str:
    """把消息列表摊成一段纯文本，给摘要器 / 抽取器当输入。

    工具结果要截断：一次接口返回的几千字 JSON 对「这段对话在聊什么」几乎没有贡献，
    却能把压缩这一次调用的成本抬高一个数量级。
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == USER:
            lines.append(f"用户：{content}")
        elif role == ASSISTANT:
            calls = message.get("tool_calls") or []
            if content:
                lines.append(f"助手：{content}")
            for call in calls:
                name = getattr(call, "name", None) or (call or {}).get("name", "工具")
                lines.append(f"（助手调用了工具 {name}）")
        elif role == TOOL:
            body = content if len(content) <= tool_limit else content[:tool_limit] + "……"
            lines.append(f"（工具结果：{body}）")
    return "\n".join(lines)


class Summarizer:
    """滚动压缩：把「已有摘要 + 新挤出的对话」交给模型，压成一段新摘要。

    提示词在 :mod:`agent.prompt`（本项目所有给模型看的文字都在那儿）。
    失败一律返回已有摘要——**宁可摘要旧一点，也不能因为一次网络抖动把历史全丢了**。
    """

    def __init__(self, llm: Any, max_chars: int = 300) -> None:
        self.llm = llm
        self.max_chars = max(80, int(max_chars))

    def compress(self, previous: str, messages: list[Message]) -> str:
        transcript = render_transcript(messages)
        if not transcript.strip():
            return previous

        prompt = SUMMARY_USER.format(
            previous=previous.strip() or "（还没有摘要，这是第一次压缩）",
            transcript=transcript,
            limit=self.max_chars,
        )
        try:
            response = self.llm.chat([system_message(SUMMARY_SYSTEM), user_message(prompt)], None)
        except Exception:
            logger.exception("压缩历史失败，沿用已有摘要")
            return previous

        summary = str(getattr(response, "content", "") or "").strip()
        if not summary:
            logger.warning("压缩返回了空摘要，沿用已有摘要")
            return previous
        # 模型经常不听「多少字以内」，这里硬截一刀。宁可断在半句上，
        # 也不能让摘要自己长成一个新的上下文负担。
        if len(summary) <= self.max_chars:
            return summary
        return summary[: self.max_chars].rstrip() + "……"


#: 「这句话里可能有值得长期记住的东西」的高精度信号。
#:
#: 这是**成本过滤，不是判据**。真正的判据是模型——这里只是先挡掉明显没什么可记的轮次，
#: 省下一次调用。所以措辞刻意放宽：漏掉一条只是少记一条，而放宽的代价不过是多一次
#: 很短的模型调用，它会自己返回空列表。
_PERSONAL = re.compile(
    r"记住|记一下|记下|别忘|我叫|我是|我的[^，。？!！\n]{0,8}(是|叫|在)|"
    r"我(更)?(喜欢|偏好|习惯|讨厌|不喜欢)|我用|我住|我在[^，。？!！\n]{0,8}(工作|上班|上学)|"
    r"以后都|下次都|帮我记"
)


def looks_personal(text: str) -> bool:
    """成本过滤：这句话值不值得跑一次抽取。"""
    return bool(_PERSONAL.search(text or ""))


class FactExtractor:
    """从一轮对话里抽出「稳定事实」。

    抽什么、不抽什么全写在 :data:`agent.prompt.FACT_SYSTEM` 里——那是提示词，
    属于需要反复调的东西，不该散在业务代码里。

    解析一律防御式：模型返回的是一段文字，可能裹着代码围栏、可能前后带解释、
    可能干脆是一句「没有需要记住的内容」。**任何解析失败都当成「这次没抽到」**，
    绝不抛异常——记忆抽取是主流程之外的一步，它失败了不该影响用户拿到答案。
    """

    def __init__(self, llm: Any, max_facts: int = 3) -> None:
        self.llm = llm
        self.max_facts = max(1, int(max_facts))

    def extract(
        self, question: str, answer: str, known: Iterable[str] = ()
    ) -> list[tuple[str, str]]:
        """返回 ``[(内容, 分类)]``。空列表表示这一轮没有值得长期记住的东西。"""
        prompt = FACT_USER.format(
            known="\n".join(f"- {item}" for item in known) or "（还没有任何记录）",
            question=question.strip()[:2000],
            answer=answer.strip()[:2000],
            limit=self.max_facts,
        )
        try:
            response = self.llm.chat([system_message(FACT_SYSTEM), user_message(prompt)], None)
        except Exception:
            logger.exception("抽取长期记忆失败，本轮跳过")
            return []
        return parse_facts(str(getattr(response, "content", "") or ""), self.max_facts)


#: 抽取结果里允许出现的分类。模型写了别的词就归到 other。
_FACT_CATEGORIES = ("identity", "preference", "project", "other")

#: 「没什么可记的」这类回答本身不是一条事实。
#:
#: 模型经常不老老实实回 `[]`，而是回一句「没有需要记住的内容」——要是把这句
#: 当事实存下来，它会出现在记忆管理界面里，而且**从此每一轮对话都被注入一次**。
#: 一条垃圾记忆比漏记一条贵得多，所以这里宁可错杀。
_NOTHING = re.compile(r"没有|暂无|无[^，。]{0,4}(可|需要)|不需要|未发现")


def _is_nothing(content: str) -> bool:
    if len(content) > 30:
        return False
    return bool(_NOTHING.search(content)) and bool(re.search(r"记|保存|提取|记录", content))


def parse_facts(raw: str, limit: int = 3) -> list[tuple[str, str]]:
    """把模型返回的一段文字解析成事实列表。解析不出来就返回空列表。

    只认 JSON：先剥代码围栏，再退一步从整段文字里抠出第一个 ``[...]``。
    **不做「一行一条」的纯文本兜底**——试过，代价太高：模型说一句
    「没有需要记住的内容」，那一整句就成了事实。要求 JSON 而模型不给 JSON 时，
    正确的做法是这次不记，而不是去猜它想说什么。
    """
    import json

    text = raw.strip()
    if text.startswith("```"):
        # ```json ... ``` / ``` ... ```
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text:
        return []

    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("["), text.rfind("]")
        if 0 <= start < end:
            try:
                data = json.loads(text[start : end + 1])
            except ValueError:
                data = None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    facts: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, str):
            content, category = item, "other"
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("fact") or "").strip()
            category = str(item.get("category") or "other").strip().lower()
        else:
            continue
        content = " ".join(content.split())
        if not content or len(content) > 200 or _is_nothing(content):
            continue
        if category not in _FACT_CATEGORIES:
            category = "other"
        facts.append((content, category))
        if len(facts) >= limit:
            break
    return facts


# --------------------------------------------------------------------------- #
# 消息切分（裁剪的基石）
# --------------------------------------------------------------------------- #


def split_into_groups(messages: Iterable[Message]) -> list[Group]:
    """把消息切成「调用组」：assistant 的工具调用 + 紧随其后的 tool 结果算一组。"""
    groups: list[Group] = []
    for message in messages:
        previous = groups[-1] if groups else None
        if (
            message.get("role") == TOOL
            and previous
            and previous[0].get("role") == ASSISTANT
            and previous[0].get("tool_calls")
        ):
            previous.append(message)
        else:
            groups.append([message])
    return groups


def split_into_turns(messages: list[Message]) -> list[Turn]:
    """把调用组归入「轮」：一条 user 消息开启一轮，直到下一条 user 消息为止。"""
    turns: list[Turn] = []
    for group in split_into_groups(messages):
        if group[0].get("role") == USER or not turns:
            turns.append([group])
        else:
            turns[-1].append(group)
    return turns


# --------------------------------------------------------------------------- #
# 记忆
# --------------------------------------------------------------------------- #


class Memory:
    """对话记忆：窗口 + 摘要 + 长期事实。

    对外只暴露「追加 / 读取 / 清空 / 压缩」几个动作，裁剪在读取时自动发生。
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_turns: int = 20,
        max_context_chars: int = 60000,
        compressor: Compressor | None = None,
        facts: Iterable[str] | None = None,
        summary: str = "",
        summary_upto: int = 0,
    ) -> None:
        self.system_prompt = system_prompt or ""
        self.max_turns = max(1, int(max_turns))
        #: 历史字符数上限（粗粒度估算，不引第三方 tokenizer；字符数 ≈ token 数的 1~2 倍）
        self.max_context_chars = max(0, int(max_context_chars))
        self.compressor = compressor
        #: 长期事实（只有正文，来源和分类在 store 里）。由会话层在每轮开始前刷新。
        self.facts: list[str] = [item for item in (facts or []) if item]
        #: 滚动摘要，以及它覆盖到 ``_messages`` 的第几条（水位线）
        self.summary = summary or ""
        self._summary_upto = max(0, int(summary_upto))
        self._messages: list[dict[str, Any]] = []
        self.trim_count = 0

    # ---------- 追加 ----------

    def add_user(self, text: str) -> dict[str, Any]:
        return self._append(user_message(text))

    def add_assistant(
        self,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        raw: Any = None,
        provider: str = "",
    ) -> dict[str, Any]:
        return self._append(assistant_message(content, tool_calls, raw=raw, provider=provider))

    def add_tool_result(
        self, tool_call_id: str, name: str, content: str, is_error: bool = False
    ) -> dict[str, Any]:
        return self._append(tool_message(tool_call_id, name, content, is_error))

    def _append(self, message: dict[str, Any]) -> dict[str, Any]:
        self._messages.append(message)
        return message

    # ---------- 长期记忆 ----------

    def set_facts(self, facts: Iterable[str]) -> None:
        """换掉注入的长期事实。会话层在记忆被增删改时调它，下一轮立刻生效。"""
        self.facts = [item for item in facts if item]

    # ---------- 读取 ----------

    def get_messages(self) -> list[dict[str, Any]]:
        """返回给模型的消息列表：system + 背景 + （裁剪后的）历史。

        裁剪只在这里发生，所以调用方拿到的永远是一份合法的历史。
        **这个方法不调用模型、不产生费用**（见模块文档）。
        """
        history, _ = self._window()
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(system_message(self.system_prompt))
        block = self.context_block()
        if block:
            messages.append(user_message(block))
        messages.extend(history)
        return messages

    def context_block(self) -> str:
        """长期事实 + 摘要拼成的那段背景。

        走 **user 消息**而不是塞进 system 提示，是刻意的：摘要的内容来自对话，
        而对话内容是不可信输入（用户可以让模型去读一个网页，网页里写什么都有可能）。
        放进 system 就等于把不可信内容抬到指令层级——那正是 OWASP 说的
        Context Injection。放在 user 层，它和「用户说的话」同级，指令层级不受污染。

        （连续两条 user 消息是合法的：Anthropic 的 Messages API 明确写了
        "Consecutive user or assistant turns in your request will be combined into a
        single turn"，OpenAI 兼容那一家子也一样。）
        """
        parts: list[str] = []
        if self.facts:
            parts.append(MEMORY_HEADER + "\n" + "\n".join(f"- {item}" for item in self.facts))
        if self.summary:
            parts.append(SUMMARY_HEADER + "\n" + self.summary)
        if not parts:
            return ""
        parts.append(CONTEXT_FOOTER)
        return "\n\n".join(parts)

    def raw_messages(self) -> list[dict[str, Any]]:
        """未经裁剪的完整历史（落盘 / 调试用）。"""
        return list(self._messages)

    # ---------- 清空与恢复 ----------

    def clear(self) -> None:
        """保留 system，清掉其余（§三.3「清空当前上下文」）。

        **摘要和长期事实不在这里被删。** 摘要是这次会话自己的，跟着一起清；
        长期事实存在 store 里、跨会话有效，属于「记忆管理」那个界面的管辖范围——
        清空一次对话不该顺手抹掉「用户是谁」。
        """
        self._messages.clear()
        self.summary = ""
        self._summary_upto = 0
        self.trim_count = 0
        logger.debug("上下文已清空（system 提示与长期记忆保留）")

    def restore(
        self, messages: Iterable[Message], summary: str = "", summary_upto: int = 0
    ) -> None:
        """把落盘的历史装回来（切换会话 / 重启后打开旧会话时用）。"""
        self._messages = [dict(message) for message in messages]
        self.summary = summary or ""
        self._summary_upto = max(0, int(summary_upto))
        self.trim_count = 0

    # ---------- 裁剪 ----------

    def _window(self) -> tuple[list[Message], int]:
        """算出「保留下来的历史」和「从前头丢掉了多少条」。

        丢弃永远以**整轮**为单位。返回值第二个是丢掉的条数，它同时也是
        保留下来的第一条消息在 ``_messages`` 里的下标——摘要的水位线就是拿它做基准的，
        所以这两个数必须同源算出来，不能各算各的。
        """
        if not self._messages:
            return [], 0

        turns = split_into_turns(self._messages)
        retained = list(turns)
        dropped: list[Turn] = []

        # 1) 按轮数裁剪
        while len(retained) > self.max_turns:
            dropped.append(retained.pop(0))

        # 2) 按字符预算裁剪（从最旧的整轮开始丢，永远保留最后一轮 = 当前问题）
        if self.max_context_chars:
            while len(retained) > 1 and self._count_chars(retained) > self.max_context_chars:
                dropped.append(retained.pop(0))

        history: list[Message] = [m for turn in retained for group in turn for m in group]

        # 3) 兜底：消息列表必须以 user 开头（各家 API 的硬要求）
        lead = 0
        while lead < len(history) and history[lead].get("role") != USER:
            lead += 1
        history = history[lead:]

        dropped_count = sum(len(group) for turn in dropped for group in turn) + lead
        if dropped_count:
            self.trim_count += len(dropped) + (1 if lead else 0)
            logger.debug("上下文超限：丢掉前 %d 条消息，保留 %d 条", dropped_count, len(history))
        return history, dropped_count

    @staticmethod
    def _count_chars(turns: list[Turn]) -> int:
        total = 0
        for turn in turns:
            for group in turn:
                for message in group:
                    total += len(str(message.get("content") or ""))
                    for call in message.get("tool_calls") or []:
                        total += len(str(call.arguments))
        return total

    # ---------- 压缩 ----------

    @property
    def summary_upto(self) -> int:
        """摘要覆盖到 ``_messages`` 的第几条（水位线）。

        单独开一个属性而不是让调用方去翻 ``stats()``：stats 要顺带算一遍各层
        token 估算，而落盘摘要时只需要这一个整数。
        """
        return self._summary_upto

    def window_start(self) -> int:
        """当前窗口的起点在 ``_messages`` 里的下标。"""
        return self._window()[1]

    def pending_messages(self) -> list[Message]:
        """已经掉出窗口、但还没进摘要的那段历史。

        这就是「该压缩了」的候选：既不在窗口里（模型看不到），又不在摘要里
        （永远丢了）。它是压缩唯一该吃的输入。
        """
        start = min(self._summary_upto, len(self._messages))
        end = self.window_start()
        return self._messages[start:end] if end > start else []

    def pending_tokens(self) -> int:
        return estimate_tokens(render_transcript(self.pending_messages()))

    def needs_compression(self, threshold: int) -> bool:
        return self.compressor is not None and self.pending_tokens() >= max(1, threshold)

    def compress(self) -> str:
        """把待压缩的那段历史滚进摘要。返回摘要正文（没动就返回原值）。

        由入口层在**一轮对话结束之后**调用——放在这一层之外是刻意的：
        它要花一次模型调用，放在 ``get_messages()`` 里会让每一次提问都多等几秒，
        而用户看到的只是「界面卡住了」。
        """
        if self.compressor is None:
            return self.summary
        pending = self.pending_messages()
        if not pending:
            return self.summary
        try:
            summary = self.compressor.compress(self.summary, pending)
        except Exception:  # 压缩失败不该影响主流程，退化成「这次不压」
            logger.exception("历史压缩失败，本次跳过")
            return self.summary

        summary = (summary or "").strip()
        if not summary or summary == self.summary:
            return self.summary
        self.summary = summary
        self._summary_upto = min(self._summary_upto, len(self._messages)) + len(pending)
        logger.info(
            "已把 %d 条历史压进摘要（累计覆盖到第 %d 条）", len(pending), self._summary_upto
        )
        return self.summary

    # ---------- 观测 ----------

    def layers(self) -> list[dict[str, Any]]:
        """上下文各层的 token 估算，给界面上那张条形图用。

        ``recent`` 用**裁剪后**的窗口算，所以这张表加起来就是下一次请求真正会发出去的量。
        """
        history, _ = self._window()
        return [
            {"key": "system", "label": "系统提示", "tokens": estimate_tokens(self.system_prompt)},
            {"key": "facts", "label": "长期记忆", "tokens": estimate_tokens("\n".join(self.facts))},
            {"key": "summary", "label": "历史摘要", "tokens": estimate_tokens(self.summary)},
            {
                "key": "recent",
                "label": "最近对话",
                "tokens": estimate_tokens(render_transcript(history, tool_limit=10**9)),
            },
        ]

    def stats(self) -> dict[str, Any]:
        turns = split_into_turns(self._messages)
        history, dropped = self._window()
        return {
            "messages": len(self._messages),
            "turns": len(turns),
            "chars": self._count_chars(turns),
            "trimmed_turns": self.trim_count,
            # 下面几个是三层记忆的观测值，界面上的上下文状态读它们
            "window_messages": len(history),
            "dropped_messages": dropped,
            "summary_upto": self.summary_upto,
            "pending_tokens": self.pending_tokens(),
            "layers": self.layers(),
            "tokens": sum(layer["tokens"] for layer in self.layers()),
            # 「窗口用掉了多少」那个比例条的分子分母。**单位是字符**：真正决定
            # 窗口裁到哪里的就是 max_context_chars，token 只是给显示用的估算。
            # 拿 tokens 去比 max_context_chars 会得到一个凭空造出来的百分比。
            "window_chars": self._count_chars(split_into_turns(history)),
            "budget_chars": self.max_context_chars,
        }

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:  # pragma: no cover
        info = self.stats()
        return (
            f"<Memory messages={info['messages']} turns={info['turns']} "
            f"tokens≈{info['tokens']} facts={len(self.facts)}>"
        )
