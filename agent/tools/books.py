"""图书检索（Open Library，免 Key）。

Open Library 是 Internet Archive 的目录，几千万条书目，**中英文都能搜**——
这是它比豆瓣/Google Books 更适合当工具的地方（前者没有公开接口，后者要 Key）。

查到的是**书目**不是全文，也不代表哪里能借到；链接给到 Open Library 的条目页，
想看详情自己去点。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, one_line, tool_timeout

logger = logging.getLogger(__name__)

_SEARCH = "https://openlibrary.org/search.json"
_ITEM = "https://openlibrary.org"

#: 只取要用的字段：不写 fields 的话一条书目能有好几 KB，八成都用不上
_FIELDS = "title,author_name,first_publish_year,key,edition_count,language,isbn,publisher,subject"


class BookSearchTool(BaseTool):
    name = "book_search"
    description = (
        "按书名、作者或主题检索图书，返回书名、作者、初版年份、出版社和 Open Library 链接。\n"
        "**用户问「有没有讲 X 的书」「某人写过什么书」「这本书哪年出的」时用它**，"
        "不要凭记忆报出版年份和作者——这类信息记混的概率很高。\n"
        "查的是书目信息，不含全文，也不代表能借到。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索词：书名、作者名或主题词，中英文都行",
            },
            "limit": {
                "type": "integer",
                "description": "返回条数，1-20，默认 5",
            },
        },
        "required": ["query"],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _term(text: str) -> str:
        """短查询要加 ``title:`` 前缀。

        Open Library 的 ``q`` 要求至少 3 个字符，而**中文书名常常只有两三个字**
        （「三体」「活着」「围城」），直接问会被 422 顶回来。加个字段前缀既能过校验，
        又顺带把范围收到书名上——实测「三体」查不到，``title:三体`` 第一条就是刘慈欣。
        """
        return text if len(text) >= 3 else f"title:{text}"

    def run(self, query: str = "", limit: int = 5) -> str:
        text = (query or "").strip()
        if not text:
            raise ToolError("检索词不能为空，给个书名、作者或主题")

        count = min(max(int(limit or 5), 1), 20)
        try:
            data = get_json(
                _SEARCH,
                params={"q": self._term(text), "limit": count, "fields": _FIELDS},
                timeout=self.timeout,
                service="Open Library",
            )
        except ToolError as exc:
            if "422" in str(exc):
                raise ToolError(
                    f"Open Library 不接受「{text}」这样的检索词（它要求至少 3 个字符，"
                    "而且多个词之间是「同时满足」）。书名 + 作者一起写，或者换个长一点的词。"
                ) from exc
            raise
        docs = data.get("docs") or []
        if not docs:
            raise ToolError(
                f"Open Library 里没有同时满足「{text}」的书。它把多个检索词当「都要有」，"
                "少给几个词（只写书名，或只写作者）命中率更高。"
            )

        total = data.get("numFound", len(docs))
        lines = [f"Open Library 检索「{text}」：命中约 {total} 条，列出前 {len(docs)} 条", ""]
        for index, doc in enumerate(docs, 1):
            authors = "、".join((doc.get("author_name") or [])[:3]) or "作者不详"
            publishers = doc.get("publisher") or []
            subjects = [one_line(s, 12) for s in (doc.get("subject") or [])[:4]]
            lines.append(
                "\n".join(
                    part
                    for part in (
                        f"{index}. {one_line(doc.get('title'))}",
                        f"   {authors} · {doc.get('first_publish_year') or '年份不详'}"
                        + (f" · {one_line(publishers[0], 40)}" if publishers else "")
                        + f" · {doc.get('edition_count', 0)} 个版本",
                        f"   主题：{'、'.join(subjects)}" if subjects else "",
                        f"   {_ITEM}{doc.get('key', '')}",
                    )
                    if part.strip()
                )
            )
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [BookSearchTool(timeout=tool_timeout(config))]
