"""加密货币行情（CoinGecko 公开接口，免 Key）。

只做「查现价」这一件事。CoinGecko 的免费额度按分钟限流，
所以这里**一次请求查多币**（``simple/price`` 支持 ``ids=bitcoin,ethereum``），
而不是一个币一次——模型问三种币，就该只花一次额度。
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, number, tool_timeout

logger = logging.getLogger(__name__)

_BASE = "https://api.coingecko.com/api/v3"
_SEARCH = f"{_BASE}/search"
_PRICE = f"{_BASE}/simple/price"

#: 中文名 → CoinGecko id。英文名和代号不用进表：``/search`` 自己认得出
#: （「btc」「eth」「bitcoin」都能搜到），进表反而要多维护一份会过期的清单。
_ALIASES = {
    "比特币": "bitcoin",
    "以太坊": "ethereum",
    "以太币": "ethereum",
    "泰达币": "tether",
    "美元硬币": "usd-coin",
    "币安币": "binancecoin",
    "索拉纳": "solana",
    "瑞波币": "ripple",
    "狗狗币": "dogecoin",
    "艾达币": "cardano",
    "波卡": "polkadot",
    "莱特币": "litecoin",
    "波场": "tron",
    "链环": "chainlink",
    "雪崩": "avalanche-2",
    "马蹄": "matic-network",
    "屎币": "shiba-inu",
    "门罗币": "monero",
    "柚子币": "eos",
    "恒星币": "stellar",
}

#: 计价货币也认中文
_CURRENCY_ALIASES = {
    "美元": "usd",
    "人民币": "cny",
    "欧元": "eur",
    "日元": "jpy",
    "英镑": "gbp",
    "港币": "hkd",
    "港元": "hkd",
    "韩元": "krw",
    "新台币": "twd",
    "澳元": "aud",
    "加元": "cad",
    "新加坡元": "sgd",
    "印度卢比": "inr",
    "卢布": "rub",
    "比特币": "btc",
    "以太坊": "eth",
}


class CryptoPriceTool(BaseTool):
    name = "crypto_price"
    description = (
        "查加密货币的当前价格和 24 小时涨跌（数据来自 CoinGecko）。\n"
        "**用户问「比特币现在多少钱」「ETH 涨了没」时用它**，不要凭记忆报价——"
        "币价一天能差 10%。\n"
        "可以一次问多个币（用逗号分隔），中文名（比特币）、英文名（bitcoin）"
        "和代号（btc）都认。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "coins": {
                "type": "string",
                "description": "币种，多个用逗号分隔，如「比特币,以太坊」或「btc,eth」",
            },
            "currency": {
                "type": "string",
                "description": "计价货币，默认 usd；常用 cny、eur、jpy",
            },
        },
        "required": ["coins"],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, coins: str = "", currency: str = "usd") -> str:
        wanted = [part.strip() for part in (coins or "").replace("，", ",").split(",")]
        wanted = [part for part in wanted if part]
        if not wanted:
            raise ToolError("币种不能为空，比如「比特币」或「btc,eth」")
        if len(wanted) > 10:
            raise ToolError(f"一次最多查 10 个币，收到 {len(wanted)} 个")

        unit = self._resolve_currency(currency)
        ids = [self._resolve_coin(name) for name in wanted]

        data = get_json(
            _PRICE,
            params={
                "ids": ",".join(ids),
                "vs_currencies": unit,
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
            timeout=self.timeout,
            service="CoinGecko",
        )
        if not isinstance(data, dict) or not data:
            raise ToolError(f"CoinGecko 没有返回 {'、'.join(wanted)} 的行情，换个币名试试")

        return self._format(wanted, ids, data, unit)

    # ---------- 参数解析 ----------

    def _resolve_coin(self, name: str) -> str:
        alias = _ALIASES.get(name)
        if alias:
            return alias

        found = get_json(
            _SEARCH,
            params={"query": name},
            timeout=self.timeout,
            service="CoinGecko 币种检索",
        )
        candidates = found.get("coins") or []
        if not candidates:
            raise ToolError(f"CoinGecko 里找不到「{name}」这个币，换个名字或直接给代号（btc）")

        lowered = name.lower()
        for item in candidates:  # 名字/代号完全对上的优先
            if lowered in ((item.get("id") or "").lower(), (item.get("symbol") or "").lower()):
                return item["id"]
        # 否则取市值排名最靠前的那个（列表本身就是按相关度+市值排的）
        ranked = sorted(candidates, key=lambda c: c.get("market_cap_rank") or 10**9)
        return ranked[0]["id"]

    @staticmethod
    def _resolve_currency(raw: str) -> str:
        text = (raw or "usd").strip().lower()
        return _CURRENCY_ALIASES.get(text, text)

    # ---------- 排版 ----------

    @staticmethod
    def _format(wanted: list[str], ids: list[str], data: dict[str, Any], unit: str) -> str:
        lines = [f"CoinGecko 行情（计价：{unit.upper()}）", ""]
        for asked, coin_id in zip(wanted, ids, strict=True):
            item = data.get(coin_id)
            if not isinstance(item, dict) or unit not in item:
                lines.append(f"{asked}（{coin_id}）：没有 {unit.upper()} 报价")
                continue

            change = item.get(f"{unit}_24h_change")
            trend = ""
            if isinstance(change, (int, float)):
                trend = f"　24h {'+' if change >= 0 else ''}{change:.2f}%"
            stamp = item.get("last_updated_at")
            updated = (
                dt.datetime.fromtimestamp(stamp).strftime("　更新 %Y-%m-%d %H:%M")
                if isinstance(stamp, (int, float))
                else ""
            )
            lines.append(
                f"{asked}（{coin_id}）　{unit.upper()} {number(float(item[unit]))}{trend}{updated}"
            )
        return "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [CryptoPriceTool(timeout=tool_timeout(config))]
