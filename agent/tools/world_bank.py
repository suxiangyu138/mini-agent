"""世界银行公开数据（免 Key）。

它是「某国某指标历年是多少」这类问题的**权威**来源——GDP、人口、失业率、城镇化率……
都比让模型凭记忆报数字可靠得多，而模型的记忆在国别统计上错得很有信心。

指标用**白名单**而不是全库检索：世界银行的指标目录有近 3 万条、拉一次 11MB，
为了「什么指标都能查」付这个代价不值。这里挑了二十来个最常问的，
认中文关键词也认官方代码；不在表里的就明确告诉模型去查 ``data.worldbank.org``。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, number, tool_timeout

logger = logging.getLogger(__name__)

_BASE = "https://api.worldbank.org/v2"
_COUNTRIES = f"{_BASE}/country"

#: 中文国名 → ISO3。世界银行的国名只有英文（``locale=zh`` 也不翻译，实测过），
#: 所以这一层翻译得自己做。表里是问得最多的那二十来个，其余用 ISO 代码或英文名。
_COUNTRY_ALIASES = {
    "中国": "CHN",
    "中国大陆": "CHN",
    "美国": "USA",
    "日本": "JPN",
    "德国": "DEU",
    "英国": "GBR",
    "法国": "FRA",
    "印度": "IND",
    "巴西": "BRA",
    "俄罗斯": "RUS",
    "韩国": "KOR",
    "意大利": "ITA",
    "加拿大": "CAN",
    "澳大利亚": "AUS",
    "西班牙": "ESP",
    "墨西哥": "MEX",
    "印度尼西亚": "IDN",
    "沙特阿拉伯": "SAU",
    "南非": "ZAF",
    "土耳其": "TUR",
    "阿根廷": "ARG",
    "越南": "VNM",
    "泰国": "THA",
    "新加坡": "SGP",
    "瑞士": "CHE",
    "荷兰": "NLD",
    "瑞典": "SWE",
    "波兰": "POL",
    "埃及": "EGY",
    "尼日利亚": "NGA",
    "中国香港": "HKG",
    "香港": "HKG",
    "中国台湾": "TWN",
    "台湾": "TWN",
}

#: 关键词 → (官方代码, 口径说明)。键统一小写后比对。
_INDICATORS: dict[str, tuple[str, str]] = {
    "gdp": ("NY.GDP.MKTP.CD", "GDP（现价美元）"),
    "国内生产总值": ("NY.GDP.MKTP.CD", "GDP（现价美元）"),
    "生产总值": ("NY.GDP.MKTP.CD", "GDP（现价美元）"),
    "gdp增速": ("NY.GDP.MKTP.KD.ZG", "GDP 年增长率（%）"),
    "gdp增长率": ("NY.GDP.MKTP.KD.ZG", "GDP 年增长率（%）"),
    "经济增长": ("NY.GDP.MKTP.KD.ZG", "GDP 年增长率（%）"),
    "人均gdp": ("NY.GDP.PCAP.CD", "人均 GDP（现价美元）"),
    "人均国内生产总值": ("NY.GDP.PCAP.CD", "人均 GDP（现价美元）"),
    "人口": ("SP.POP.TOTL", "总人口"),
    "总人口": ("SP.POP.TOTL", "总人口"),
    "人口增长率": ("SP.POP.GROW", "人口年增长率（%）"),
    "预期寿命": ("SP.DYN.LE00.IN", "出生时预期寿命（岁）"),
    "人均寿命": ("SP.DYN.LE00.IN", "出生时预期寿命（岁）"),
    "失业率": ("SL.UEM.TOTL.ZS", "失业率（占劳动力 %，ILO 估算）"),
    "通胀": ("FP.CPI.TOTL.ZG", "通货膨胀率（CPI 年增 %）"),
    "通货膨胀": ("FP.CPI.TOTL.ZG", "通货膨胀率（CPI 年增 %）"),
    "cpi": ("FP.CPI.TOTL.ZG", "通货膨胀率（CPI 年增 %）"),
    "碳排放": ("EN.GHG.CO2.MT.CE.AR5", "二氧化碳排放总量（百万吨 CO2e）"),
    "二氧化碳": ("EN.GHG.CO2.MT.CE.AR5", "二氧化碳排放总量（百万吨 CO2e）"),
    "互联网普及率": ("IT.NET.USER.ZS", "互联网使用人口占比（%）"),
    "网民": ("IT.NET.USER.ZS", "互联网使用人口占比（%）"),
    "城镇化率": ("SP.URB.TOTL.IN.ZS", "城镇人口占比（%）"),
    "城市人口": ("SP.URB.TOTL.IN.ZS", "城镇人口占比（%）"),
    "教育支出": ("SE.XPD.TOTL.GD.ZS", "政府教育支出（占 GDP %）"),
    "医疗支出": ("SH.XPD.CHEX.GD.ZS", "经常性卫生支出（占 GDP %）"),
    "卫生支出": ("SH.XPD.CHEX.GD.ZS", "经常性卫生支出（占 GDP %）"),
    "生育率": ("SP.DYN.TFRT.IN", "总和生育率（每名妇女生育数）"),
    "识字率": ("SE.ADT.LITR.ZS", "成人识字率（%）"),
    "出口占gdp": ("NE.EXP.GNFS.ZS", "商品与服务出口（占 GDP %）"),
    "制造业占比": ("NV.IND.MANF.ZS", "制造业增加值（占 GDP %）"),
    "人均国民收入": ("NY.GNP.PCAP.CD", "人均 GNI（Atlas 法，现价美元）"),
    "gni": ("NY.GNP.PCAP.CD", "人均 GNI（Atlas 法，现价美元）"),
    "面积": ("AG.SRF.TOTL.K2", "国土面积（平方公里）"),
    "国土面积": ("AG.SRF.TOTL.K2", "国土面积（平方公里）"),
    "军费": ("MS.MIL.XPND.GD.ZS", "军费开支（占 GDP %）"),
    "国防开支": ("MS.MIL.XPND.GD.ZS", "军费开支（占 GDP %）"),
}

_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{2,}$")
_RANGE_PATTERN = re.compile(r"^(\d{4})\s*[:\-~]\s*(\d{4})$")
_YEAR_PATTERN = re.compile(r"^\d{4}$")


class WorldBankTool(BaseTool):
    name = "world_bank"
    description = (
        "查世界银行的国别统计数据：GDP、人均 GDP、GDP 增速、人口、失业率、通胀、"
        "预期寿命、城镇化率、碳排放、互联网普及率等，可查单个国家也可多国对比。\n"
        "**用户问「某国某年的 GDP/人口/失业率是多少」「中日美对比」这类**"
        "**国别统计数字时用它**，不要凭记忆报数字——记忆里的国别数据经常是错的或过期的。\n"
        "数据来自世界银行，通常有 1 年左右的滞后，最新年份可能是空的。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "国家，中文名（中国）、英文名（China）或 ISO 代码（CHN/CN）都行。"
                "多国对比用逗号分隔，如「中国,美国,日本」",
            },
            "indicator": {
                "type": "string",
                "description": "指标，中文关键词（GDP、人口、失业率、通胀、城镇化率…）"
                "或官方代码（NY.GDP.MKTP.CD）",
            },
            "years": {
                "type": "string",
                "description": "年份：区间「2015:2023」、单年「2023」，默认最近 5 年",
            },
        },
        "required": ["country", "indicator"],
    }

    #: 国别清单（400 条，约 40KB）拉一次就够，进程内缓存
    _country_table: dict[str, str] | None = None

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, country: str = "", indicator: str = "", years: str = "") -> str:
        codes, labels = self._resolve_countries(country or "")
        code, name = self._resolve_indicator(indicator or "")
        window = self._window(years)

        data = get_json(
            f"{_COUNTRIES}/{';'.join(codes)}/indicator/{code}",
            params={"format": "json", "per_page": 500, "date": window},
            timeout=self.timeout,
            service="世界银行接口",
        )
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            raise ToolError(
                f"世界银行没有「{'、'.join(labels)}」的「{name or code}」"
                f"在 {window} 区间的数据。可能这门统计这个国家不报，换个年份或换个指标试试。"
            )

        return self._format(name, code, dict(zip(codes, labels, strict=True)), data[1], window)

    # ---------- 参数解析 ----------

    def _resolve_countries(self, raw: str) -> tuple[list[str], list[str]]:
        wanted = [part.strip() for part in raw.replace("，", ",").split(",") if part.strip()]
        if not wanted:
            raise ToolError("国家不能为空")

        table = self._load_country_table()
        codes: list[str] = []
        labels: list[str] = []
        for item in wanted:
            key = item.lower()
            code = _COUNTRY_ALIASES.get(item) or table.get(key)
            if not code and _CODE_PATTERN.match(item):
                code = item.upper()  # 直接给的代码，交给 API 去认
            if not code:
                raise ToolError(
                    f"认不出国家「{item}」。中文名、英文名（China）或 ISO 代码（CHN）都行；"
                    "中文名只覆盖了常见的三十来个，其余的请用英文名或代码。"
                )
            codes.append(code)
            labels.append(item if item.upper() != code else code)

        if len(codes) > 8:
            raise ToolError(f"一次最多比 8 个国家，收到 {len(codes)} 个")
        return codes, labels

    def _load_country_table(self) -> dict[str, str]:
        """ISO2/ISO3/英文名 → ISO3。查不到就退化成空表（只认代码和中文别名）。"""
        if WorldBankTool._country_table is not None:
            return WorldBankTool._country_table
        try:
            data = get_json(
                _COUNTRIES,
                params={"format": "json", "per_page": 400},
                timeout=self.timeout,
                service="世界银行国别清单",
            )
            table: dict[str, str] = {}
            for item in (data[1] if isinstance(data, list) and len(data) > 1 else []) or []:
                iso3 = (item.get("id") or "").upper()
                if not iso3:
                    continue
                table[iso3.lower()] = iso3
                iso2 = (item.get("iso2Code") or "").upper()
                if iso2:
                    table[iso2.lower()] = iso3
                name = (item.get("name") or "").strip()
                if name:
                    table[name.lower()] = iso3
            WorldBankTool._country_table = table
            return table
        except ToolError as exc:
            logger.debug("拿不到世界银行国别清单，只认代码和中文别名：%s", exc)
            WorldBankTool._country_table = {}
            return {}

    @staticmethod
    def _resolve_indicator(raw: str) -> tuple[str, str]:
        text = raw.strip()
        if not text:
            raise ToolError("指标不能为空，比如 GDP、人口、失业率")

        alias = _INDICATORS.get(text.lower())
        if alias:
            return alias

        upper = text.upper()
        if "." in upper and _CODE_PATTERN.match(upper):
            return upper, ""  # 已经是官方代码了，排版时就不必把代码再念一遍

        keywords = "、".join(sorted({name for _, name in _INDICATORS.values()}))
        raise ToolError(
            f"内置指标表里没有「{text}」。可用的有：{keywords}。"
            f"也可以直接传世界银行官方代码（形如 NY.GDP.MKTP.CD），"
            f"完整目录见 https://data.worldbank.org/indicator"
        )

    @staticmethod
    def _window(years: str) -> str:
        text = (years or "").strip()
        if _RANGE_PATTERN.match(text):
            start, end = _RANGE_PATTERN.match(text).groups()  # type: ignore[union-attr]
            return f"{start}:{end}"
        if _YEAR_PATTERN.match(text):
            return text
        this_year = dt.date.today().year
        return f"{this_year - 5}:{this_year}"

    # ---------- 排版 ----------

    @staticmethod
    def _format(
        name: str,
        code: str,
        labels: dict[str, str],
        rows: list[dict[str, Any]],
        window: str,
    ) -> str:
        # API 按国家分组返回，这里保持分组、组内按年份排好
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            country = row.get("country") or {}
            # 用问的人自己给的名字（「中国」），而不是 API 回的英文名（「China」）。
            # 注意 API 的 country.id 是**两位**码（CN），三位码在 countryiso3code 里。
            key = (
                labels.get((row.get("countryiso3code") or "").upper())
                or labels.get((country.get("id") or "").upper())
                or country.get("value")
                or "?"
            )
            grouped.setdefault(key, []).append(row)

        title = f"{name}（{code}）" if name else code
        lines = [f"{title}，区间 {window}"]
        for key, items in grouped.items():
            items.sort(key=lambda r: r.get("date") or "", reverse=True)
            real = [r for r in items if r.get("value") is not None]
            if not real:
                lines.append(f"\n{key}：这个区间没有数据（世界银行统计通常滞后 1 年左右）")
                continue
            lines.append(f"\n{key}：")
            for row in real:
                lines.append(f"  {row['date']}  {number(float(row['value']))}")
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [WorldBankTool(timeout=tool_timeout(config))]
