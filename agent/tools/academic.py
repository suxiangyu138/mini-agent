"""学术检索（OpenAlex / Crossref / PubMed，都是免 Key 的）。

三个库合在一个工具里，靠 ``source`` 参数切换，而不是拆成三个工具：
模型挑得动「用哪个库」，但每次都带上三份 Schema 是白花的 token，
而且它经常三个都想试——一个工具换参数比三个工具轮着调省好几步。

分工（写进 description 里让它自己选）：
- ``openalex``：覆盖最广、带引用数，**默认**
- ``crossref``：DOI 元数据最权威，找「这篇的正式出处」用它
- ``pubmed``：生物医学专库，医学问题用它
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, one_line, tool_timeout

logger = logging.getLogger(__name__)

_OPENALEX = "https://api.openalex.org/works"
_CROSSREF = "https://api.crossref.org/works"
_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_SOURCES = ("openalex", "crossref", "pubmed")


class AcademicSearchTool(BaseTool):
    name = "academic_search"
    description = (
        "检索学术文献（论文、综述），返回标题、作者、年份、发表venue、引用数和 DOI。\n"
        "**用户问「有没有相关研究」「找几篇论文」「这个领域的进展」时用它**，"
        "不要凭记忆报论文标题——记忆里的标题和作者经常是错的。\n"
        "source 三选一：openalex（默认，覆盖最广、带引用数）、"
        "crossref（DOI 元数据最权威）、pubmed（生物医学专库，医学问题用它）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索词。英文关键词召回率明显更高，中文词也能查但结果较少",
            },
            "source": {
                "type": "string",
                "enum": list(_SOURCES),
                "description": "用哪个库，默认 openalex",
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

    def run(self, query: str = "", source: str = "openalex", limit: int = 5) -> str:
        text = (query or "").strip()
        if not text:
            raise ToolError("检索词不能为空")

        which = (source or "openalex").strip().lower()
        if which not in _SOURCES:
            raise ToolError(f"source 只能是 {'/'.join(_SOURCES)}，收到「{source}」")

        count = min(max(int(limit or 5), 1), 20)
        if which == "openalex":
            return self._openalex(text, count)
        if which == "crossref":
            return self._crossref(text, count)
        return self._pubmed(text, count)

    # ---------- OpenAlex ----------

    def _openalex(self, query: str, limit: int) -> str:
        data = get_json(
            _OPENALEX,
            params={"search": query, "per-page": limit},
            timeout=self.timeout,
            service="OpenAlex",
        )
        works = data.get("results") or []
        if not works:
            return f"OpenAlex 里没搜到「{query}」。换个英文关键词，或放宽检索词试试。"

        total = (data.get("meta") or {}).get("count", len(works))
        lines = [f"OpenAlex 检索「{query}」：命中约 {total} 篇，列出前 {len(works)} 篇", ""]
        for index, work in enumerate(works, 1):
            source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            link = (work.get("doi") or work.get("id") or "").replace("https://doi.org/", "doi:")
            lines.append(
                "\n".join(
                    part
                    for part in (
                        f"{index}. {one_line(work.get('title') or work.get('display_name'))}",
                        f"   {self._names(work.get('authorships') or [])}",
                        f"   {work.get('publication_year') or '年份不详'}"
                        f" · {source or 'venue 不详'}"
                        f" · 被引 {work.get('cited_by_count', 0)}"
                        f" · {work.get('type') or ''}",
                        f"   {link}",
                    )
                    if part.strip()
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _names(authorships: list[dict[str, Any]], keep: int = 4) -> str:
        names = [(item.get("author") or {}).get("display_name") for item in authorships[:keep]]
        names = [n for n in names if n]
        if not names:
            return "作者不详"
        more = len(authorships) - len(names)
        return "、".join(names) + (f" 等 {len(authorships)} 人" if more > 0 else "")

    # ---------- Crossref ----------

    def _crossref(self, query: str, limit: int) -> str:
        data = get_json(
            _CROSSREF,
            params={
                "query": query,
                "rows": limit,
                # 只要这几个字段：Crossref 的完整记录动辄几十 KB，没必要全拉回来
                "select": "DOI,title,author,issued,container-title,URL,is-referenced-by-count,type",
            },
            timeout=self.timeout,
            service="Crossref",
        )
        items = (data.get("message") or {}).get("items") or []
        if not items:
            return f"Crossref 里没搜到「{query}」。换个英文关键词试试。"

        total = (data.get("message") or {}).get("total-results", len(items))
        lines = [f"Crossref 检索「{query}」：命中约 {total} 条，列出前 {len(items)} 条", ""]
        for index, item in enumerate(items, 1):
            authors = item.get("author") or []
            names = [
                one_line(" ".join(filter(None, (a.get("given"), a.get("family")))))
                for a in authors[:4]
            ]
            names = [n for n in names if n]
            year = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
            lines.append(
                "\n".join(
                    part
                    for part in (
                        f"{index}. {one_line((item.get('title') or ['(无标题)'])[0])}",
                        f"   {'、'.join(names) or '作者不详'}"
                        + (f" 等 {len(authors)} 人" if len(authors) > len(names) else ""),
                        f"   {year or '年份不详'} · "
                        f"{(item.get('container-title') or ['venue 不详'])[0]}"
                        f" · 被引 {item.get('is-referenced-by-count', 0)}"
                        f" · {item.get('type') or ''}",
                        f"   doi:{item.get('DOI') or ''}",
                    )
                    if part.strip()
                )
            )
        return "\n".join(lines)

    # ---------- PubMed ----------

    def _pubmed(self, query: str, limit: int) -> str:
        found = get_json(
            _PUBMED_SEARCH,
            params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
            timeout=self.timeout,
            service="PubMed 检索",
        )
        result = found.get("esearchresult") or {}
        ids = result.get("idlist") or []
        if not ids:
            return (
                f"PubMed 里没搜到「{query}」。换个英文医学关键词，"
                "或者用 MeSH 词（如 «diabetes mellitus, type 2»）试试。"
            )

        # esearch 只给 PMID，标题摘要得再问一次 esummary —— 两次往返是 PubMed 的固定姿势
        detail = get_json(
            _PUBMED_SUMMARY,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=self.timeout,
            service="PubMed 详情",
        )
        records = detail.get("result") or {}
        total = result.get("count", len(ids))

        lines = [f"PubMed 检索「{query}」：命中约 {total} 条，列出前 {len(ids)} 条", ""]
        for index, pmid in enumerate(ids, 1):
            item = records.get(pmid) or {}
            authors = [a.get("name") for a in (item.get("authors") or [])[:4] if a.get("name")]
            doi = next(
                (
                    a.get("value")
                    for a in (item.get("articleids") or [])
                    if a.get("idtype") == "doi"
                ),
                "",
            )
            venue = " ".join(
                filter(None, (item.get("source"), item.get("volume"), item.get("pages")))
            )
            lines.append(
                "\n".join(
                    part
                    for part in (
                        f"{index}. {one_line(item.get('title'), 200) or '(无标题)'}",
                        f"   {'、'.join(authors) or '作者不详'}",
                        f"   {item.get('pubdate') or '日期不详'} · {venue or '期刊不详'}",
                        f"   PMID {pmid}" + (f" · doi:{doi}" if doi else ""),
                    )
                    if part.strip()
                )
            )
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [AcademicSearchTool(timeout=tool_timeout(config))]
