"""Hacker News 热榜（Firebase 公开接口，免 Key）。

取一条榜单要 1 + N 次请求：``topstories`` 只给一串 id，每条内容都得单独取。
所以这里**并发取**——串行 8 条要两三秒，并发一轮就回来了。

HN 的价值不在「新闻」而在**技术圈此刻在讨论什么**：
模型答不出「最近有什么值得看的」，但这个榜单答得出。
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

from .base import BaseTool, ToolError
from .net import get_json, one_line, tool_timeout

logger = logging.getLogger(__name__)

_BASE = "https://hacker-news.firebaseio.com/v0"
_DISCUSS = "https://news.ycombinator.com/item?id="

#: 榜单名 → 接口路径片段
_LISTS = {
    "top": "topstories",
    "new": "newstories",
    "best": "beststories",
    "ask": "askstories",
    "show": "showstories",
    "job": "jobstories",
}


class HackerNewsTool(BaseTool):
    name = "hacker_news"
    description = (
        "看 Hacker News 热榜，返回标题、得分、评论数和原文链接。\n"
        "**用户问「技术圈最近在聊什么」「有什么值得看的」「最近有什么新东西」时用它**；"
        "它反映的是此刻技术社区的关注点，不是新闻通稿。\n"
        "board 可选 top（综合热榜，默认）/ new（最新）/ best（高分）/ "
        "ask（Ask HN 讨论）/ show（Show HN 作品）/ job（招聘）。"
    )
    parameters = {
        #: 参数名不叫 ``list``：那会把内置的 ``list`` 挡在这个方法里，
        #: 底下的 ``isinstance(ids, list)`` 和 ``list(pool.map(...))`` 会一起失效。
        "type": "object",
        "properties": {
            "board": {
                "type": "string",
                "enum": list(_LISTS),
                "description": "看哪个榜，默认 top",
            },
            "count": {
                "type": "integer",
                "description": "返回条数，1-20，默认 8",
            },
        },
        "required": [],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, board: str = "top", count: int = 8) -> str:
        which = (board or "top").strip().lower()
        if which not in _LISTS:
            raise ToolError(f"board 只能是 {'/'.join(_LISTS)}，收到「{board}」")

        limit = min(max(int(count or 8), 1), 20)
        ids = get_json(
            f"{_BASE}/{_LISTS[which]}.json",
            timeout=self.timeout,
            service="Hacker News 榜单",
        )
        if not isinstance(ids, list) or not ids:
            raise ToolError("Hacker News 榜单是空的，稍后再试")

        wanted = ids[:limit]
        # 榜单顺序就是排名顺序；并发取但结果按原顺序摆回来
        with ThreadPoolExecutor(max_workers=min(len(wanted), 8)) as pool:
            items = list(pool.map(self._item, wanted))

        lines = [f"Hacker News「{which}」榜前 {len(wanted)} 条", ""]
        for index, item in enumerate(items, 1):
            if not item:
                continue
            title = one_line(item.get("title") or item.get("text"), 160) or "(无标题)"
            link = item.get("url") or f"{_DISCUSS}{item.get('id')}"
            host = urlsplit(item["url"]).hostname or "" if item.get("url") else ""
            when = item.get("time")
            stamp = (
                dt.datetime.fromtimestamp(when).strftime("%m-%d %H:%M")
                if isinstance(when, (int, float))
                else ""
            )
            lines.append(
                f"{index}. {title}\n"
                f"   {item.get('score', 0)} 分 · {item.get('descendants', 0)} 评论 · "
                f"{item.get('by', '')} · {stamp}" + (f" · {host}" if host else "") + f"\n   {link}"
            )
        return "\n".join(lines)

    def _item(self, item_id: Any) -> dict[str, Any] | None:
        """单条取不到就跳过——一条挂了不该让整份榜单失败。"""
        try:
            data = get_json(
                f"{_BASE}/item/{item_id}.json",
                timeout=self.timeout,
                service="Hacker News 详情",
            )
            return data if isinstance(data, dict) else None
        except ToolError as exc:
            logger.debug("HN 条目 %s 取不到：%s", item_id, exc)
            return None


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [HackerNewsTool(timeout=tool_timeout(config))]
