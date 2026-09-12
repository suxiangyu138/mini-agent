"""时间 / 日期查询工具。

模型没有「现在几点」的概念（训练数据有截止时间），所以必须给它一个查时间的工具，
否则它会拿训练数据里的日期硬编。支持 IANA 时区名（Asia/Shanghai、UTC、America/New_York）。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _resolve_tz(name: str):
    """把时区名解析成 tzinfo。local 走本机时区，其余按 IANA 名解析。"""
    key = (name or "local").strip()
    if key.lower() in ("local", "本地", ""):
        return datetime.now().astimezone().tzinfo, "本地时区"
    if key.upper() == "UTC":
        return timezone.utc, "UTC"
    try:
        return ZoneInfo(key), key
    except ZoneInfoNotFoundError as exc:
        raise ToolError(
            f"未知时区 {name!r}。请用 IANA 时区名，例如 Asia/Shanghai、UTC、America/New_York。"
            "（Windows 上若报错，需要 pip install tzdata）"
        ) from exc
    except ValueError as exc:
        raise ToolError(f"时区名 {name!r} 不合法：{exc}") from exc


class CurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = (
        "获取当前日期和时间。**任何涉及「今天」「现在」「今年」「还有几天」的问题，都必须先调用它**，"
        "不要依赖你训练数据里的时间。\n"
        "返回指定时区的完整日期、时间、星期和 Unix 时间戳。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA 时区名，如 Asia/Shanghai、UTC、America/New_York。"
                    "不确定用户所在地时用 local（本机时区）。默认 local。"
                ),
            }
        },
        "required": [],
    }

    def run(self, timezone: str = "local") -> str:
        tz, label = _resolve_tz(timezone)
        now = datetime.now(tz)
        return (
            f"当前时间（{label}）：\n"
            f"  日期：{now.strftime('%Y-%m-%d')}（{_WEEKDAYS[now.weekday()]}）\n"
            f"  时间：{now.strftime('%H:%M:%S')}\n"
            f"  ISO 8601：{now.isoformat()}\n"
            f"  Unix 时间戳：{int(now.timestamp())}\n"
            f"  时区偏移：UTC{now.strftime('%z')}"
        )


class DateDiffTool(BaseTool):
    name = "date_diff"
    description = (
        "计算两个日期之间相差多少天/秒，或者在某个日期上加减天数得到新日期。\n"
        "两种用法：\n"
        "  1) 同时给 start 和 end：算出两者间隔（用于「还有几天到期」「入职多久了」）。\n"
        "  2) 只给 start 和 days：在 start 上加上 days 天（days 可以为负），算出新日期。\n"
        "日期格式统一用 YYYY-MM-DD。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "起始日期，格式 YYYY-MM-DD"},
            "end": {"type": "string", "description": "结束日期，格式 YYYY-MM-DD。与 days 二选一"},
            "days": {"type": "integer", "description": "要加/减的天数（可为负）。与 end 二选一"},
        },
        "required": ["start"],
    }

    def run(self, start: str = "", end: str = "", days: int | None = None) -> str:
        start_date = self._parse(start, "start")

        if end:
            end_date = self._parse(end, "end")
            delta = end_date - start_date
            total_days = delta.days
            total_seconds = int(delta.total_seconds())
            return (
                f"{start} → {end}：\n"
                f"  相差 {total_days} 天（{total_seconds} 秒）\n"
                f"  约 {total_days / 7:.2f} 周 / {total_days / 30.44:.2f} 个月"
                f" / {total_days / 365.25:.2f} 年"
            )

        if days is not None:
            result = start_date + timedelta(days=days)
            return (
                f"{start} {'+' if days >= 0 else '-'} {abs(days)} 天 = "
                f"{result.strftime('%Y-%m-%d')}（{_WEEKDAYS[result.weekday()]}）"
            )

        raise ToolError("请至少提供 end（算间隔）或 days（算偏移）中的一个")

    @staticmethod
    def _parse(text: str, field: str):
        value = (text or "").strip()
        if not value:
            raise ToolError(f"参数 {field} 不能为空，格式 YYYY-MM-DD")
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        raise ToolError(f"参数 {field}={text!r} 不是合法日期，请用 YYYY-MM-DD 格式")


def build_tools(config: Any = None) -> list[BaseTool]:
    """工厂函数：签名与其它工具模块保持一致，方便注册中心批量装配。"""
    del config  # 时间工具不依赖配置，参数只为统一调用方式
    return [CurrentTimeTool(), DateDiffTool()]
