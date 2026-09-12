"""公共假期查询（Nager.Date，免 Key）。

「今年还剩哪些假」「国庆放几天」「明年春节是几号」——这类问题的答案
**每年都在变**（调休、农历），模型背下来的日期基本不可信，必须查。

Nager.Date 覆盖 100 多个国家（含中国、美国、日本、德国……），
给的是法定假日本身；中国的**调休上班日**它不管，所以拿到结果后
要提醒模型「具体以国务院办公厅通知为准」。
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, tool_timeout

logger = logging.getLogger(__name__)

_BASE = "https://date.nager.at/api/v3"

#: 中文国名 → ISO 3166-1 alpha-2。只收常问的，其余给英文名或两位代码。
_ALIASES = {
    "中国": "CN",
    "中国大陆": "CN",
    "美国": "US",
    "日本": "JP",
    "德国": "DE",
    "英国": "GB",
    "法国": "FR",
    "意大利": "IT",
    "西班牙": "ES",
    "加拿大": "CA",
    "澳大利亚": "AU",
    "新西兰": "NZ",
    "韩国": "KR",
    "新加坡": "SG",
    "马来西亚": "MY",
    "泰国": "TH",
    "印度": "IN",
    "巴西": "BR",
    "墨西哥": "MX",
    "俄罗斯": "RU",
    "荷兰": "NL",
    "瑞士": "CH",
    "瑞典": "SE",
    "挪威": "NO",
    "丹麦": "DK",
    "波兰": "PL",
    "奥地利": "AT",
    "爱尔兰": "IE",
    "葡萄牙": "PT",
    "希腊": "GR",
    "土耳其": "TR",
    "南非": "ZA",
    "阿根廷": "AR",
    "中国香港": "HK",
    "香港": "HK",
    "中国台湾": "TW",
    "台湾": "TW",
}


class HolidaysTool(BaseTool):
    name = "holidays"
    description = (
        "查某个国家某年的法定公共假期，返回日期、当地名称和英文名称。\n"
        "**用户问「今年还有哪些假」「某国几号放假」「XX 节是哪天」时用它**，"
        "不要凭记忆报日期——法定假日每年都在调，中国的还涉及农历和调休。\n"
        "注意：返回的是法定假日本身，**不含调休上班安排**，"
        "所以中国的情况要补一句「具体以国务院办公厅通知为准」。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "国家，中文名（中国）、英文名（Japan）或两位代码（JP）",
            },
            "year": {
                "type": "integer",
                "description": "年份，如 2026；默认今年",
            },
            "upcoming_only": {
                "type": "boolean",
                "description": "只看今天及以后的假期，默认 false（给全年）",
            },
        },
        "required": ["country"],
    }

    #: 支持的国家清单（约 200 条）拉一次缓存住
    _countries: dict[str, str] | None = None

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, country: str = "", year: int = 0, upcoming_only: bool = False) -> str:
        code, label = self._resolve(country or "")
        today = dt.date.today()
        target_year = int(year or today.year)
        if not 1900 <= target_year <= 2100:
            raise ToolError(f"年份 {target_year} 看着不像话，给个四位年份吧")

        data = get_json(
            f"{_BASE}/PublicHolidays/{target_year}/{code}",
            timeout=self.timeout,
            service="Nager.Date 假期接口",
        )
        if not isinstance(data, list) or not data:
            raise ToolError(
                f"Nager.Date 没有 {label}（{code}）{target_year} 年的数据——"
                "这个国家可能不在它的覆盖范围里，或者年份太远"
            )

        days = [item for item in data if isinstance(item, dict) and item.get("date")]
        if upcoming_only:
            cutoff = today.isoformat()
            days = [item for item in days if item["date"] >= cutoff]
            if not days:
                return f"{label} {target_year} 年从 {cutoff} 起没有剩下的法定假期了。"

        days.sort(key=lambda item: item["date"])
        return self._format(label, code, target_year, days, upcoming_only)

    # ---------- 参数解析 ----------

    def _resolve(self, raw: str) -> tuple[str, str]:
        text = raw.strip()
        if not text:
            raise ToolError("国家不能为空，比如「中国」")
        alias = _ALIASES.get(text)
        if alias:
            return alias, text
        if len(text) == 2 and text.isalpha():
            return text.upper(), text.upper()

        table = self._load_countries()
        code = table.get(text.lower())
        if code:
            return code, text
        raise ToolError(
            f"认不出国家「{text}」。用两位代码（CN、JP、DE）或英文名；"
            "中文名只覆盖了常见的三十来个。"
        )

    def _load_countries(self) -> dict[str, str]:
        if HolidaysTool._countries is not None:
            return HolidaysTool._countries
        try:
            data = get_json(
                f"{_BASE}/AvailableCountries",
                timeout=self.timeout,
                service="Nager.Date 国别清单",
            )
            table: dict[str, str] = {}
            for item in data if isinstance(data, list) else []:
                name = (item.get("name") or "").strip()
                code = (item.get("countryCode") or "").strip().upper()
                if name and code:
                    table[name.lower()] = code
            HolidaysTool._countries = table
            return table
        except ToolError as exc:
            logger.debug("拿不到 Nager.Date 国别清单，只认代码和中文别名：%s", exc)
            HolidaysTool._countries = {}
            return {}

    # ---------- 排版 ----------

    @staticmethod
    def _format(
        label: str, code: str, year: int, days: list[dict[str, Any]], upcoming_only: bool
    ) -> str:
        today = dt.date.today()
        scope = "今天起剩余" if upcoming_only else "全年"
        lines = [f"{label}（{code}）{year} 年法定公共假期，共 {len(days)} 天（{scope}）", ""]
        for item in days:
            try:
                date = dt.date.fromisoformat(item["date"])
                weekday = "一二三四五六日"[date.weekday()]
                left = (date - today).days
            except ValueError:
                weekday, left = "", 0

            when = f"（周{weekday}）" if weekday else ""
            countdown = ""
            if left > 0:
                countdown = f"　还有 {left} 天"
            elif left == 0:
                countdown = "　就是今天"

            names = [n for n in (item.get("localName"), item.get("name")) if n]
            types = [t for t in (item.get("types") or []) if t]
            lines.append(
                f"  {item['date']}{when}  {' / '.join(dict.fromkeys(names))}"
                + (f"　[{'、'.join(types)}]" if types else "")
                + countdown
            )

        if code == "CN":
            lines.append("\n（不含调休上班安排，具体以国务院办公厅通知为准）")
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [HolidaysTool(timeout=tool_timeout(config))]
