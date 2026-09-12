"""公共 API 工具共用的取数小工具。

**这个模块不是工具模块**（没有 ``build_tools``），所以不会出现在 ``_BUILDERS`` 里，
它的东西只会被别的工具模块 import。

为什么不复用 ``http_request``
----------------------------
``http_request`` 的 URL 是**模型给的**，所以它必须扛住 SSRF 那一套：IP 字面量、私网段、
allowlist（见 :mod:`agent.tools.http`）。而这里所有 URL 的主机名都在代码里写死，
模型只能挑参数，**没有任何办法把请求指到别的主机上**——安全边界因此从「校验 URL」
换成了「主机名不进参数」，后者比 allowlist 更硬：allowlist 是补丁，这里是没那个自由度。

也正因为主机固定，这些工具不需要用户去配 ``http_allowed_hosts``，
在开假 IP 的代理（Clash / Surge）下也不会被误伤。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ToolError

logger = logging.getLogger(__name__)

#: 有些公共 API 会对没有 UA 的请求直接掐断，带上自己是谁比较礼貌。
USER_AGENT = "mini-agent/0.1 (+https://github.com/suxiangyu138/mini-agent)"

#: 这些接口都是小 JSON，15 秒不回就没必要等了——让模型早点换个参数重试，
#: 比攥着 60 秒（``config.timeout`` 的默认值）把整轮对话卡死强。
DEFAULT_TIMEOUT = 15.0
MAX_TIMEOUT = 20.0


def tool_timeout(config: Any = None) -> float:
    """取超时：跟随 ``config.timeout``，但封顶 :data:`MAX_TIMEOUT`。"""
    raw = getattr(config, "timeout", None)
    try:
        return min(float(raw), MAX_TIMEOUT) if raw else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    service: str = "接口",
) -> Any:
    """GET 一个 JSON。**所有**失败都转成 :class:`ToolError`，不往上抛别的异常。

    带上 ``service`` 是为了让报错能说清「是谁挂了」——模型看到
    「Open-Meteo 响应超时」就知道该等一会儿还是该换个城市名。
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ToolError("未安装 requests，无法访问公共 API：pip install requests") from exc

    try:
        response = requests.get(
            url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
    except requests.Timeout as exc:
        raise ToolError(f"{service} 响应超时（{timeout:.0f} 秒），稍后重试或换个查询") from exc
    except requests.RequestException as exc:
        raise ToolError(f"{service} 请求失败：{type(exc).__name__}: {exc}") from exc

    if response.status_code == 429:
        raise ToolError(f"{service} 触发限流（429），等一会儿再试，或减少调用次数")
    if response.status_code == 404:
        raise ToolError(f"{service} 没有这条记录（404）——检查一下名字或编号是否写对")
    if response.status_code >= 400:
        raise ToolError(f"{service} 返回 {response.status_code}：{response.text[:200]}")

    try:
        return response.json()
    except ValueError as exc:
        raise ToolError(f"{service} 返回的不是 JSON：{response.text[:200]}") from exc


def one_line(text: Any, limit: int = 0) -> str:
    """把一段可能带换行的文本压成一行，供列表里展示。"""
    flat = " ".join(str(text or "").split())
    if limit and len(flat) > limit:
        return flat[:limit] + "……"
    return flat


def number(value: float) -> str:
    """数字按大小选合适的小数位——1421 亿写成 1421000000000.0 没人想看。"""
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return str(value)
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if magnitude == 0:
        return "0"
    return f"{value:.4f}".rstrip("0").rstrip(".")
