"""天气预报（Open-Meteo，免注册、免 Key）。

为什么是 Open-Meteo 而不是 wttr.in：wttr.in 返回的是**给人看的纯文本**，
拿来做工具还得再解析一遍；Open-Meteo 直接给结构化 JSON，而且自带地名解析
（``geocoding-api``），省掉一张「城市名 → 经纬度」的表——那张表是这类工具
最容易烂掉的地方。

两个主机都在这里写死，模型只能给城市名（见 :mod:`agent.tools.net` 的说明）。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, number, tool_timeout

logger = logging.getLogger(__name__)

_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST = "https://api.open-meteo.com/v1/forecast"

#: WMO 天气代码 → 中文。Open-Meteo 只给数字，翻译是这一层的事。
_WMO = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴小冰雹",
    99: "雷阵雨伴大冰雹",
}


class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "查询某个地方的当前天气和未来几天预报，数据来自 Open-Meteo。\n"
        "**用户问「今天冷不冷」「明天要带伞吗」「周末天气怎么样」「要不要穿外套」时用它**，"
        "不要凭训练数据里的印象猜天气。\n"
        "地点直接用中文城市名（如「北京」「杭州」），国外城市写英文名更准（如「Tokyo」）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "地名，如「北京」「上海」「San Francisco」",
            },
            "days": {
                "type": "integer",
                "description": "预报天数，1-7，默认 3（今天算第一天）",
            },
        },
        "required": ["location"],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, location: str = "", days: int = 3) -> str:
        name = (location or "").strip()
        if not name:
            raise ToolError("地点不能为空，给个城市名，比如「北京」")

        place = self._locate(name)
        forecast = self._forecast(place, days)
        return self._format(place, forecast, days)

    # ---------- 地名 → 经纬度 ----------

    def _locate(self, name: str) -> dict[str, Any]:
        data = get_json(
            _GEOCODE,
            params={"name": name, "count": 1, "language": "zh", "format": "json"},
            timeout=self.timeout,
            service="Open-Meteo 地名解析",
        )
        results = data.get("results") or []
        if not results:
            raise ToolError(
                f"没找到叫「{name}」的地方。换个说法试试："
                "中文城市名直接用（「杭州」），国外城市用英文（「Paris」），"
                "或者把省/州一起写上（「Springfield, Illinois」）。"
            )
        return results[0]

    # ---------- 取预报 ----------

    def _forecast(self, place: dict[str, Any], days: int) -> dict[str, Any]:
        count = min(max(int(days or 3), 1), 7)
        return get_json(
            _FORECAST,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max,sunrise,sunset",
                "timezone": "auto",  # 用当地时区，不然「今天」会错一天
                "forecast_days": count,
            },
            timeout=self.timeout,
            service="Open-Meteo 天气接口",
        )

    # ---------- 排版 ----------

    @staticmethod
    def _format(place: dict[str, Any], data: dict[str, Any], days: int) -> str:
        where = " ".join(
            part for part in (place.get("country"), place.get("admin1"), place.get("name")) if part
        )
        lines = [f"{where}（纬度 {place['latitude']:.2f}，经度 {place['longitude']:.2f}）"]

        current = data.get("current") or {}
        if current:
            code = _WMO.get(current.get("weather_code"), f"未知天气({current.get('weather_code')})")
            lines.append(
                f"实况（{current.get('time', '')}）：{code}，"
                f"{number(current.get('temperature_2m', 0))}°C"
                f"（体感 {number(current.get('apparent_temperature', 0))}°C），"
                f"湿度 {current.get('relative_humidity_2m', '-')}%，"
                f"风速 {number(current.get('wind_speed_10m', 0))} km/h，"
                f"降水 {number(current.get('precipitation', 0))} mm"
            )

        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        if not dates:
            lines.append("\n（没有拿到预报数据）")
            return "\n".join(lines)

        lines.append(f"\n未来 {min(len(dates), max(int(days or 3), 1))} 天预报：")
        for index, date in enumerate(dates):
            code = _WMO.get(
                (daily.get("weather_code") or [None])[index],
                f"未知({(daily.get('weather_code') or [None])[index]})",
            )
            lines.append(
                f"  {date}  {code}  "
                f"{number((daily.get('temperature_2m_min') or [0])[index])}"
                f"~{number((daily.get('temperature_2m_max') or [0])[index])}°C  "
                f"降水 {number((daily.get('precipitation_sum') or [0])[index])}mm"
                f"（概率 {(daily.get('precipitation_probability_max') or ['-'])[index]}%）  "
                f"日出 {(daily.get('sunrise') or [''])[index][-5:]} "
                f"日落 {(daily.get('sunset') or [''])[index][-5:]}"
            )
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [WeatherTool(timeout=tool_timeout(config))]
