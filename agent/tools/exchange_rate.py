"""汇率换算（Frankfurter 主用，open.er-api 兜底，都免 Key）。

两个源是有分工的，不是随便挑一个：

- **Frankfurter**（欧洲央行数据）覆盖约 30 种主流货币，但**支持历史日期**——
  「去年这时候汇率多少」这种问题只有它能答；
- **open.er-api** 覆盖 160 多种货币，但只有最新价——人民币换泰铢这类
  Frankfurter 不管的货币对，靠它兜底。

先问 Frankfurter 支不支持这两个币，支持就用它（能带历史），
不支持才落到 er-api。查询失败绝不静默给个错数——汇率错一位就是钱的事。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, number, tool_timeout

logger = logging.getLogger(__name__)

_FRANKFURTER = "https://api.frankfurter.app"
_ER_API = "https://open.er-api.com/v6/latest"

#: 中文币名 → ISO 4217。只收常用的，其余让模型给代码（三位字母它记得住）。
_ALIASES = {
    "人民币": "CNY",
    "元": "CNY",
    "美元": "USD",
    "美金": "USD",
    "欧元": "EUR",
    "日元": "JPY",
    "日币": "JPY",
    "英镑": "GBP",
    "港币": "HKD",
    "港元": "HKD",
    "新台币": "TWD",
    "台币": "TWD",
    "韩元": "KRW",
    "澳元": "AUD",
    "澳大利亚元": "AUD",
    "加元": "CAD",
    "加拿大元": "CAD",
    "新加坡元": "SGD",
    "新币": "SGD",
    "瑞士法郎": "CHF",
    "卢布": "RUB",
    "印度卢比": "INR",
    "卢比": "INR",
    "泰铢": "THB",
    "越南盾": "VND",
    "马来西亚林吉特": "MYR",
    "马币": "MYR",
    "印尼盾": "IDR",
    "菲律宾比索": "PHP",
    "新西兰元": "NZD",
    "瑞典克朗": "SEK",
    "挪威克朗": "NOK",
    "丹麦克朗": "DKK",
    "波兰兹罗提": "PLN",
    "土耳其里拉": "TRY",
    "墨西哥比索": "MXN",
    "巴西雷亚尔": "BRL",
    "南非兰特": "ZAR",
    "沙特里亚尔": "SAR",
    "阿联酋迪拉姆": "AED",
    "以色列谢克尔": "ILS",
    "捷克克朗": "CZK",
    "匈牙利福林": "HUF",
    "罗马尼亚列伊": "RON",
    "保加利亚列弗": "BGN",
    "冰岛克朗": "ISK",
    "黄金": "XAU",
}

_CODE_PATTERN = re.compile(r"^[A-Za-z]{3}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ExchangeRateTool(BaseTool):
    name = "exchange_rate"
    description = (
        "查汇率并换算金额，支持历史日期。\n"
        "**用户问「X 币等于多少 Y 币」「100 美元换多少人民币」「上个月那天汇率多少」时用它**，"
        "不要凭记忆报汇率——汇率每天都在动，记忆里的数字一定是错的。\n"
        "常见货币可用中文名（美元、人民币、日元），其余用三位代码（THB、VND）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "from_currency": {"type": "string", "description": "源货币，如「美元」或 USD"},
            "to_currency": {
                "type": "string",
                "description": "目标货币，多个用逗号分隔（如「人民币,日元」）；"
                "省略则给一组主要货币",
            },
            "amount": {"type": "number", "description": "换算金额，默认 1"},
            "date": {
                "type": "string",
                "description": "历史日期 YYYY-MM-DD；省略则用最新汇率。"
                "只有主流货币支持历史，其余货币只能给最新价",
            },
        },
        "required": ["from_currency"],
    }

    #: Frankfurter 支持的币种清单，拉一次缓存住
    _supported: set[str] | None = None

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(
        self,
        from_currency: str = "",
        to_currency: str = "",
        amount: float = 1.0,
        date: str = "",
    ) -> str:
        base = self._resolve(from_currency, "源货币")
        targets = self._resolve_targets(to_currency, base)
        value = float(amount or 1.0)
        day = (date or "").strip()
        if day and not _DATE_PATTERN.match(day):
            raise ToolError(f"日期得是 YYYY-MM-DD 格式，收到「{day}」")

        if self._frankfurter_has(base, targets):
            return self._via_frankfurter(base, targets, value, day)
        return self._via_er_api(base, targets, value, day)

    # ---------- 参数解析 ----------

    @staticmethod
    def _resolve(raw: str, what: str) -> str:
        text = (raw or "").strip()
        if not text:
            raise ToolError(f"{what}不能为空")
        alias = _ALIASES.get(text)
        if alias:
            return alias
        if _CODE_PATTERN.match(text):
            return text.upper()
        raise ToolError(
            f"认不出{what}「{text}」。用三位货币代码（USD、CNY、THB）或常见中文名"
            f"（美元、人民币、日元、港币、泰铢…）"
        )

    def _resolve_targets(self, raw: str, base: str) -> list[str]:
        text = (raw or "").strip()
        if not text:
            # 没指定就给一组主要的，去掉源货币自己
            targets = [code for code in ("CNY", "USD", "EUR", "JPY", "GBP", "HKD") if code != base]
        else:
            targets = [
                self._resolve(part, "目标货币")
                for part in text.replace("，", ",").split(",")
                if part.strip()
            ]
        unique = list(dict.fromkeys(code for code in targets if code != base))
        if not unique:
            raise ToolError(f"源货币和目标货币都是 {base}，没法换算")
        return unique[:8]

    # ---------- 数据源 ----------

    def _frankfurter_has(self, base: str, targets: list[str]) -> bool:
        if ExchangeRateTool._supported is None:
            try:
                data = get_json(
                    f"{_FRANKFURTER}/currencies",
                    timeout=self.timeout,
                    service="Frankfurter 币种清单",
                )
                ExchangeRateTool._supported = {str(k).upper() for k in data}
            except ToolError as exc:
                logger.debug("拿不到 Frankfurter 币种清单，直接试 er-api：%s", exc)
                return False
        return base in ExchangeRateTool._supported and all(
            code in ExchangeRateTool._supported for code in targets
        )

    def _via_frankfurter(self, base: str, targets: list[str], amount: float, day: str) -> str:
        url = f"{_FRANKFURTER}/{day}" if day else f"{_FRANKFURTER}/latest"
        data = get_json(
            url,
            params={"from": base, "to": ",".join(targets)},
            timeout=self.timeout,
            service="Frankfurter 汇率",
        )
        rates = data.get("rates") or {}
        if not rates:
            raise ToolError(f"Frankfurter 没有 {base} 兑 {'、'.join(targets)} 的汇率")

        actual = data.get("date") or day
        if day and actual and actual != day:
            note = f"（{day} 不是交易日，取的是最近一个交易日 {actual}）"
        else:
            note = "（欧洲央行参考汇率）"
        return self._format(base, amount, rates, f"汇率日期 {actual} {note}")

    def _via_er_api(self, base: str, targets: list[str], amount: float, day: str) -> str:
        data = get_json(
            f"{_ER_API}/{base}",
            timeout=self.timeout,
            service="open.er-api 汇率",
        )
        if data.get("result") != "success":
            raise ToolError(
                f"open.er-api 查不到 {base} 的汇率：{data.get('error-type') or '未知错误'}"
            )

        all_rates = data.get("rates") or {}
        rates = {code: all_rates[code] for code in targets if code in all_rates}
        missing = [code for code in targets if code not in all_rates]
        if not rates:
            raise ToolError(f"两个源都查不到 {base} 兑 {'、'.join(targets)} 的汇率")

        note = f"（open.er-api，更新于 {data.get('time_last_update_utc', '未知时间')}）"
        if day:
            note += f"　注意：{base} 这类货币没有历史汇率，这里给的是最新价，不是 {day} 的"
        if missing:
            note += f"　（没有 {'、'.join(missing)} 的数据）"
        return self._format(base, amount, rates, note.strip())

    # ---------- 排版 ----------

    @staticmethod
    def _format(base: str, amount: float, rates: dict[str, Any], note: str) -> str:
        lines = [f"{number(amount)} {base} {note}", ""]
        for code, rate in rates.items():
            try:
                value = float(rate)
            except (TypeError, ValueError):
                continue
            converted = number(amount * value)
            lines.append(f"  = {converted} {code}　（1 {base} = {number(value)} {code}）")
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [ExchangeRateTool(timeout=tool_timeout(config))]
