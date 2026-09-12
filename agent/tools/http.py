"""HTTP 请求工具（可选能力）。

安全注意：这个工具请求的 URL 是**模型决定**的，所以默认禁止访问内网与保留地址
（127.0.0.1、10.x、192.168.x、169.254.169.254 云元数据端点等），
避免模型被提示词注入后拿去探测内网。确有需要时把 ``http_allow_private`` 设为 true。

**fake-ip 代理环境**（本机开着 Clash/Surge 一类代理的 fake-ip 模式）：域名会被
解析成 198.18.x.x。那是个**占位地址**，不对应任何真实主机，真实地址只有代理知道
——拿它做安全检查，检查的是一个假答案，结论只能是「每个域名都不许访问」。
所以这里对**域名解析结果全是 fake-ip 占位地址**的情况直接放行，交给代理去路由；
拿到真实地址就照旧走全套检查。于是代理开着能用，关掉之后域名解析回真实 IP，
同一套代码依旧把内网拦在外面，两边都不用改配置。

IP 字面量不受这条影响：它不经过 DNS，写什么就是什么，一律照查（见 ``_check_url``）。
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from typing import Any
from urllib.parse import urlparse

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_ALLOWED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")

#: RFC 2544 基准测试保留段。本机代理软件的 fake-ip 模式默认从这个段里发假地址。
#: 对**域名解析结果**放行（那是占位地址，不是目的地，理由见模块文档）；
#: 对 **IP 字面量**照拦不误——字面量不经过 DNS，写什么就是什么。
_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


class HttpRequestTool(BaseTool):
    name = "http_request"
    #: 响应体是答案本身，一律整份返回，不受工具层长度上限约束。
    truncate_result = False
    description = (
        "发起一次 HTTP 请求并返回响应内容（状态码 + 响应体）。\n"
        "用于访问公开 API、获取网页原始内容（注意：返回的是 HTML 源码，不是解析后的正文）。\n"
        "**不能访问内网地址**（localhost、192.168.x、10.x 等会被拒绝）。\n"
        "响应体**原样整份返回，不做截断**：头部会报出正文字符数，嫌大就换个更精确的 URL，"
        "别在同一个大页面上反复请求。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "完整 URL，必须带 http:// 或 https://"},
            "method": {
                "type": "string",
                "enum": list(_ALLOWED_METHODS),
                "description": "HTTP 方法，默认 GET",
            },
            "headers": {
                "type": "object",
                "description": '自定义请求头，如 {"Accept": "application/json"}',
            },
            "body": {
                "type": "string",
                "description": "请求体原文（POST/PUT 时使用，通常是 JSON 字符串）",
            },
            "timeout": {"type": "integer", "description": "超时秒数，默认 15，最大 60"},
        },
        "required": ["url"],
    }

    def __init__(
        self,
        allow_private: bool = False,
        timeout: float = 15.0,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self.allow_private = allow_private
        self.timeout = timeout
        self.allowed_hosts = [h.strip().lower().lstrip(".") for h in (allowed_hosts or []) if h]

    def run(
        self,
        url: str = "",
        method: str = "GET",
        headers: dict[str, Any] | None = None,
        body: str = "",
        timeout: int | None = None,
    ) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底
            raise ToolError("未安装 requests，无法发起 HTTP 请求：pip install requests") from exc

        target = (url or "").strip()
        via_fake_ip = self._check_url(target)

        verb = (method or "GET").upper()
        if verb not in _ALLOWED_METHODS:
            raise ToolError(f"不支持的方法 {method!r}，可用：{', '.join(_ALLOWED_METHODS)}")

        seconds = min(max(float(timeout or self.timeout), 1.0), 60.0)
        payload = body.encode("utf-8") if body else None

        try:
            response = requests.request(
                verb,
                target,
                headers=headers or {},
                data=payload,
                timeout=seconds,
                allow_redirects=True,
            )
        except requests.Timeout as exc:
            raise ToolError(f"请求超时（{seconds:.0f} 秒）：{target}") from exc
        except requests.TooManyRedirects as exc:
            raise ToolError(f"重定向次数过多：{target}") from exc
        except requests.RequestException as exc:
            raise ToolError(
                f"请求失败：{type(exc).__name__}: {exc}{self._proxy_hint(via_fake_ip)}"
            ) from exc

        return self._format(response)

    # ---------- 内部 ----------

    def _check_url(self, url: str) -> bool:
        """检查这个 URL 能不能请求。

        返回「解析出来的地址是否**全是** fake-ip 占位地址」——只为在连不上的时候
        给一句像样的提示（见 ``_proxy_hint``），不参与放行判断。
        """
        parsed = urlparse(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ToolError(f"URL 必须以 http:// 或 https:// 开头，收到：{url!r}")
        host = parsed.hostname
        if not host:
            raise ToolError(f"URL 里没有主机名：{url!r}")
        if self.allow_private:
            return False

        # 主机名本身就是 IP 字面量：不经过 DNS，代理软件也改不了，所以要**先于白名单**判定。
        # 白名单是给「域名被 fake-ip 解析成假地址」这种情况用的，不该顺手把真内网地址也放进来。
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            self._reject_unsafe(literal, host)
            return False

        if self._is_allowed_host(host):
            return False

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port)
        except socket.gaierror as exc:
            raise ToolError(f"域名无法解析：{host}（{exc}）") from exc

        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue

        # 占位地址不是目的地，跳过；剩下的是真地址，照查。
        real = [ip for ip in addresses if ip not in _FAKE_IP_NETWORK]
        for ip in real:
            self._reject_unsafe(ip, host)

        if addresses and not real:
            logger.debug("%s 解析到 fake-ip 占位地址 %s，跳过保留地址检查", host, addresses[0])
            return True
        return False

    @staticmethod
    def _proxy_hint(via_fake_ip: bool) -> str:
        """连不上、而刚才解析出来的又全是 fake-ip 占位地址 —— 那多半不是网址的问题。"""
        if not via_fake_ip:
            return ""
        return (
            "（域名解析到 198.18.x.x 这个 fake-ip 占位段却没连上：多半是代理没开，"
            "或者刚关掉、DNS 缓存里还是假地址。重新解析一次通常就好）"
        )

    def _is_allowed_host(self, host: str) -> bool:
        """白名单匹配：写 ``example.com`` 同时放行 ``api.example.com``。"""
        name = host.lower().rstrip(".")
        return any(name == item or name.endswith("." + item) for item in self.allowed_hosts)

    @staticmethod
    def _reject_unsafe(ip: Any, host: str) -> None:
        if not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return
        # 走到这里还带着 198.18.x.x 的，只剩 IP 字面量一种情况：域名解析出的占位地址
        # 在 _check_url 里已经被滤掉了。字面量是写死的真地址，白名单也帮不上忙。
        hint = (
            f"（{ip} 属于 fake-ip 常用段。代理的 fake-ip 只对**域名**生效——"
            f"直接写 IP 字面量没有意义，也不会被放行）"
            if ip in _FAKE_IP_NETWORK
            else "如果确实需要，请把配置项 http_allow_private 设为 true"
        )
        raise ToolError(f"出于安全考虑，禁止访问内网/保留地址（{host} → {ip}）。{hint}。")

    @staticmethod
    def _format(response: Any) -> str:
        content_type = response.headers.get("Content-Type", "")
        try:
            text = response.text
        except Exception:  # 解码失败不该打爆工具
            text = repr(response.content[:2000])

        if "application/json" in content_type.lower():
            try:
                text = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                pass  # 声明是 JSON 但解析失败，就按原文返回

        # 正文原样返回，一个字都不切（见类属性 truncate_result）。
        # 头部只报长度：不切内容，但要让模型知道这次拿回来的是多大一坨。
        return (
            f"HTTP {response.status_code} {response.reason}\n"
            f"Content-Type: {content_type or '(未声明)'}\n"
            f"最终 URL: {response.url}\n"
            f"正文长度: {len(text)} 字符\n"
            f"{'-' * 40}\n{text}"
        )


def build_tools(config: Any = None) -> list[BaseTool]:
    allow_private = bool(getattr(config, "http_allow_private", False))
    timeout = float(getattr(config, "timeout", 15.0))
    allowed_hosts = list(getattr(config, "http_allowed_hosts", None) or [])
    return [
        HttpRequestTool(allow_private=allow_private, timeout=timeout, allowed_hosts=allowed_hosts)
    ]
