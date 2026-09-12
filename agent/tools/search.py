"""网页搜索工具（可选，需要 API Key）。

设计方案里它被标为「可选，需 API」。这里用 Tavily 的 REST 接口（返回结构化 JSON，实现最简单）。
没有配置 ``TAVILY_API_KEY`` 时，这个工具**不会被注册**——与其让模型调用后收到一句
「没配 key」，不如干脆不出现，省得它浪费一步。

想换成别的搜索服务（Serper / Bing / SearXNG）：
照着 :class:`WebSearchTool` 实现一个新的 BaseTool，然后在 ``build_tools`` 里返回它即可。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.tavily.com/search"


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "联网搜索，返回若干条搜索结果的标题、链接和摘要。\n"
        "**当问题涉及训练数据之后的事件、实时信息（今天的新闻、当前价格、最新版本号），"
        "或你不确定的事实时，用它而不是凭记忆回答。**\n"
        "搜索结果只是摘要，如果需要某个链接的完整内容，再用 http_request 抓取该链接。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词。用自然语言问句效果通常更好"},
            "max_results": {
                "type": "integer",
                "description": "返回结果条数，默认 5，范围 1-10",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str, timeout: float = 20.0, max_results: int = 5) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    def run(self, query: str = "", max_results: int | None = None) -> str:
        text = (query or "").strip()
        if not text:
            raise ToolError("搜索关键词不能为空")

        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ToolError("未安装 requests，无法联网搜索：pip install requests") from exc

        count = min(max(int(max_results or self.max_results), 1), 10)
        try:
            response = requests.post(
                _ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": text,
                    "max_results": count,
                    "search_depth": "basic",
                    "include_answer": True,
                },
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise ToolError(f"搜索超时（{self.timeout:.0f} 秒）：{text}") from exc
        except requests.RequestException as exc:
            raise ToolError(f"搜索请求失败：{type(exc).__name__}: {exc}") from exc

        if response.status_code == 401:
            raise ToolError("搜索 API Key 无效或已过期，请检查 TAVILY_API_KEY")
        if response.status_code == 429:
            raise ToolError("搜索接口触发限流（429），请稍后再试或减少调用次数")
        if response.status_code >= 400:
            raise ToolError(f"搜索接口返回 {response.status_code}：{response.text[:300]}")

        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError(f"搜索接口返回的不是 JSON：{response.text[:300]}") from exc

        return self._format(text, data)

    @staticmethod
    def _format(query: str, data: dict[str, Any]) -> str:
        lines = [f"搜索：{query}"]

        answer = (data.get("answer") or "").strip()
        if answer:
            lines.append(f"\n【摘要回答】{answer}")

        results = data.get("results") or []
        if not results:
            lines.append("\n（没有找到相关结果，试试换个关键词）")
            return "\n".join(lines)

        lines.append(f"\n【搜索结果 {len(results)} 条】")
        for index, item in enumerate(results, 1):
            title = (item.get("title") or "(无标题)").strip()
            url = item.get("url") or ""
            snippet = (item.get("content") or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "……"
            lines.append(f"\n{index}. {title}\n   {url}\n   {snippet}")
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """没有 API Key 就返回空列表——工具不注册，模型也就不会去调它。"""
    api_key = (getattr(config, "search_api_key", "") or "").strip()
    if not api_key:
        logger.debug("未配置 search_api_key，跳过 web_search 工具")
        return []
    timeout = float(getattr(config, "timeout", 20.0))
    return [WebSearchTool(api_key=api_key, timeout=timeout)]
