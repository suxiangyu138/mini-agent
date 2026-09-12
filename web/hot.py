"""主页的实时热点推送：把 Hacker News 此刻在聊的东西，变成可以直接点的问题。

放在入口层（和 ``web/server.py`` 里的 ``SUGGESTIONS`` 并列），因为它不是给模型
用的工具，是主页的内容——只不过内容是现拉的。

三条约束决定了这里的写法：

- **不能阻塞主页**。所以它是独立端点 ``/api/hot``，前端开机之后再拉，
  拉不到就返回空 groups，位置让回静态建议。主页自己永远是先出来的。
- **问句必须带来源链接**。模型手里没有 ``web_search``（那要配 API Key），
  所以每条推送都挂上原文 URL，让 ``http_request`` 真去读。
  不挂链接的话，这个功能就是在系统性地诱导模型凭记忆编造时事——
  而推送的内容按定义就在训练数据之后。
- **一个源挂了不牵连整体**。刷新失败就沿用上一次的结果并标 ``stale``；
  一次都没成功过，前端回落到静态建议。主页永远不会空。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

from agent.tools.base import ToolError
from agent.tools.net import one_line
from agent.tools.tech_news import fetch_stories

logger = logging.getLogger("web.hot")

#: 一次拉多少条 HN 榜单来挑。每条都要单独请求一次详情，30 条够挑了。
#: 减到 15 以下时两个分组都会常常凑不满——实测 30 条里大约命中 6 条。
_SCAN = 30

#: 每个分组最多留几条
_PER_GROUP = 3

#: 一共最多几条。和 ``web/server.py`` 的 ``_MAX_SUGGESTIONS`` 对齐——
#: 建议区是竖着一列排的，再多就把首屏撑满了。
_MAX_ITEMS = 5

#: 缓存有效期。HN 榜单没有分钟级的变化，一小时刷一次足够。
_TTL = 3600.0

#: 拉取超时。**主页在等它**，所以比工具那 15 秒短得多：
#: 宁可这次少推几条，也不能让人对着半张首页干等。
_TIMEOUT = 6.0

_DISCUSS = "https://news.ycombinator.com/item?id="

#: 词表写成相邻字符串字面量拼接再 split()，而不是元组字面量。
#: 理由很实际：一百多个词的元组 ruff format 会拆成一行一个、一百多行，
#: 按类分组的读法就没了。这里是**数据不是代码**，按行铺开既好读又好加词。
#: （项目里其余地方都规规矩矩走 ruff format，只有这两张表是例外。）
_AI_WORDS = (
    """
    ai a.i agi llm llms gpt chatgpt claude openai anthropic gemini llama mistral
    deepseek qwen kimi copilot model models neural transformer transformers
    diffusion rag prompt prompts agent agents agentic embedding embeddings
    inference finetune finetuning nlp multimodal hallucination gpu cuda
    tokenizer quantization
    """
    # split() 处理不了带空格的词，这两个单独接上去
    " machine learning deep learning"
).split()

_INFRA_WORDS = (
    # 存储与数据
    """
    database databases sql postgres postgresql sqlite mysql redis mongodb
    clickhouse elasticsearch kafka rabbitmq storage filesystem index query
    transaction replication sharding cluster clusters consistency
    """
    # 运行时与语言
    """
    runtime compiler interpreter kernel linux bsd rust golang erlang elixir zig
    allocator memory syscall concurrency concurrent async await threads
    """
    # 网络与服务
    """
    server servers backend api apis http tcp udp dns tls grpc graphql proxy
    socket nginx cdn
    """
    # 部署与运维
    """
    deploy deployment kubernetes k8s docker container containers serverless
    cloudflare terraform distributed cache caching queue microservice
    microservices observability profiling latency throughput scaling benchmark
    infra infrastructure
    """
).split()

#: 注意 "machine learning" / "deep learning" 是当成**一个**词条进词表的。
#: 上面那行 " machine learning deep learning" 拼出来是 "…quantization machine
#: learning deep learning"，split() 之后就是两个带空格的词条，正好。


def _matcher(words: Sequence[str]) -> re.Pattern[str]:
    """把词表编成一个「按词匹配」的正则。

    为什么不能直接 ``word in title``：``ai`` 会命中 said / email / chair，
    ``ml`` 会命中 html / xml。这类误判会直接摆到主页上，很扎眼。

    ``\\b`` 在这里也不够用：词表里有 ``a.i``、``postgresql`` 这种，边界得自己划。
    前后都用 ``[\\w.]`` 挡（点也要挡，否则 ``ai.example.com`` 这种主机名会被当成
    命中），但**不挡连字符**——挡了 ``AI-powered`` 就认不出来了。
    """
    return re.compile(
        r"(?<![\w.])(?:" + "|".join(re.escape(word) for word in words) + r")(?![\w.])",
        re.I,
    )


#: 兴趣分组。顺序即优先级：两边都命中且命中数相同时，排在前面的赢。
#: 词表刻意写长——少推一条无所谓，推一条不相干的很伤。
_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "AI / 大模型",
        "ai",
        tuple(_AI_WORDS),
    ),
    (
        "后端 / 基础设施",
        "infra",
        tuple(_INFRA_WORDS),
    ),
)

_MATCHERS = tuple(_matcher(words) for _, _, words in _GROUPS)

#: ``Show HN: xxx`` 这种前缀。剥掉它问句才读得顺，但前缀本身是信息
#: （Show HN = 有人做了个东西），所以靠它选模板，不是直接丢弃。
_PREFIX = re.compile(r"^(Ask HN|Show HN|Tell HN|Launch HN)\s*:\s*", re.I)


def _classify(title: str) -> str | None:
    """标题落到哪个分组；两边都不沾就返回 ``None``（这条不推）。

    比的是命中数：一条标题里 AI 词出现 3 次、基础设施词出现 1 次，算 AI。
    同分时**排在前面的分组赢**（严格大于才算超越），也就是算 AI——
    这是主兴趣。
    """
    best_score = 0
    best_key: str | None = None
    for key, matcher in zip((k for _, k, _ in _GROUPS), _MATCHERS, strict=True):
        score = len(matcher.findall(title))
        if score > best_score:
            best_score, best_key = score, key
    return best_key


def to_question(title: str) -> str:
    """标题 → 一句可以直接发出去的话。

    带前缀的单独套模板：Show HN 是「有人做了个东西」，Ask HN 是「有人抛了个
    问题」，都套「是怎么回事」会读着别扭。
    """
    match = _PREFIX.match(title)
    if not match:
        return f"「{title}」是怎么回事？"
    body = title[match.end() :].strip()
    if match.group(1).lower() in ("show hn", "launch hn"):
        return f"「{body}」这个项目怎么样？"
    return f"「{body}」大家是怎么讨论的？"


#: HN 的 ``url`` 字段是投稿人随手填的。``javascript:`` / ``data:`` 这类进了问句，
#: 就会被当成「来源」原样递给模型、再喂回 ``http_request``。工具层确实有一道
#: scheme 白名单（``http.py::_check_url``），但那条不变式属于工具层——
#: 本层的出口不该隔着一个模块去依赖它。所以在这里自己收干净。
_SAFE_SCHEMES = ("http://", "https://")


def link_of(story: dict[str, Any]) -> str:
    """原文链接。Ask HN 这类帖子没有 ``url``，退回 HN 讨论页。

    不是 http(s) 的一律当没有——宁可退回讨论页，也不把一段来源不明的
    scheme 当成「原文链接」交出去。
    """
    url = str(story.get("url") or "").strip()
    if url.lower().startswith(_SAFE_SCHEMES):
        return url
    if url:
        logger.info("丢弃无法识别的链接 %r，退回讨论页", url[:80])
    return f"{_DISCUSS}{story.get('id')}"


def _build() -> list[dict[str, Any]]:
    """拉一次 HN，分好组、排好序、转成问句。取不到就抛 :class:`ToolError`。"""
    stories = fetch_stories(board="top", limit=_SCAN, timeout=_TIMEOUT)

    #: 分组 → [(分数, 条目)]。分数只用来排序，不进最终输出。
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {k: [] for _, k, _ in _GROUPS}
    seen: set[str] = set()
    for story in stories:
        title = one_line(story.get("title"), 140)
        if not title:
            continue
        url = link_of(story)
        if url in seen:  # 同一篇被转贴两次的话只留先出现的那条
            continue
        key = _classify(title)
        if key is None:
            continue
        seen.add(url)
        score = story.get("score") or 0
        buckets[key].append(
            (
                score,
                {
                    "question": to_question(title),
                    "url": url,
                    "meta": f"HN · {score} 分 · {story.get('descendants') or 0} 评论",
                },
            )
        )

    groups: list[dict[str, Any]] = []
    budget = _MAX_ITEMS
    for label, key, _ in _GROUPS:
        if budget <= 0:
            break
        ranked = sorted(buckets[key], key=lambda pair: -pair[0])[: min(_PER_GROUP, budget)]
        if not ranked:  # 这组今天没料就整组不出现，不留一个空标题
            continue
        items = [item for _, item in ranked]
        budget -= len(items)
        groups.append({"label": label, "items": items})
    return groups


_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "groups": []}


def groups_with_status() -> dict[str, Any]:
    """给 ``/api/hot`` 用。**不抛异常**——首页的一个装饰性模块没有资格让接口报错。

    缓存过期就顺手刷一次；刷失败就沿用上一次的结果并标 ``stale``，
    好让界面上能说一句「可能不是最新的」。一次都没成功过则返回空 groups，
    前端会回落到 ``web/server.py`` 里那套静态建议。

    取数全程持锁：并发来几个请求也只会有一次真的去拉，
    其余的等这一把锁，出来后直接读缓存。
    """
    with _lock:
        if _cache["at"] and time.monotonic() - _cache["at"] < _TTL:
            return {"groups": _cache["groups"], "stale": False}
        try:
            groups = _build()
        except ToolError as exc:
            logger.info("热点推送取数失败，本轮不刷新：%s", exc)
            return {"groups": _cache["groups"], "stale": bool(_cache["groups"])}
        _cache["groups"] = groups
        _cache["at"] = time.monotonic()
        return {"groups": groups, "stale": False}


def warm() -> None:
    """开机时就在后台拉一次，把冷启动那几秒挪到没人看的时候。

    取数要三四秒（网络路径的固有延迟 × 轮数），要是等用户打开页面才开始拉，
    第一屏就得干等这几秒。提前拉好，之后一个 TTL 周期内都是缓存命中。

    daemon 线程：进程该退就退，不为了预热把退出拖住。失败了也无所谓——
    :func:`groups_with_status` 本来就兜着底。
    """
    threading.Thread(target=groups_with_status, name="hot-warm", daemon=True).start()
