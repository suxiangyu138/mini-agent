"""HTTP 请求工具（可选能力）。

安全注意：这个工具请求的 URL 是**模型决定**的，所以默认禁止访问内网与保留地址
（127.0.0.1、10.x、192.168.x、169.254.169.254 云元数据端点等），
避免模型被提示词注入后拿去探测内网。确有需要时把 ``http_allow_private`` 设为 true。
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

_MAX_BODY_CHARS = 8000
_ALLOWED_SCHEMES = ("http", "https")
_ALLOWED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")

#: RFC 2544 基准测试保留段。本机代理软件的 fake-ip 模式默认从这个段里发假地址，
#: 所以「域名解析到这里」基本等于「这台机器开着 fake-ip」，提示语要区别对待。
_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = (
        "发起一次 HTTP 请求并返回响应内容（状态码 + 响应体）。\n"
        "用于访问公开 API、获取网页原始内容（注意：返回的是 HTML 源码，不是解析后的正文）。\n"
        "**不能访问内网地址**（localhost、192.168.x、10.x 等会被拒绝）。\n"
        "响应体过长会截断。需要 JSON 结果时，可以配合说明让模型自己解释响应体。"
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
        self._check_url(target)

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
            raise ToolError(f"请求失败：{type(exc).__name__}: {exc}") from exc

        return self._format(response)

    # ---------- 内部 ----------

    def _check_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ToolError(f"URL 必须以 http:// 或 https:// 开头，收到：{url!r}")
        host = parsed.hostname
        if not host:
            raise ToolError(f"URL 里没有主机名：{url!r}")
        if self.allow_private:
            return

        # 主机名本身就是 IP 字面量：不经过 DNS，代理软件也改不了，所以要**先于白名单**判定。
        # 白名单是给「域名被 fake-ip 解析成假地址」这种情况用的，不该顺手把真内网地址也放进来。
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            self._reject_unsafe(literal, host)
            return

        if self._is_allowed_host(host):
            return

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port)
        except socket.gaierror as exc:
            raise ToolError(f"域名无法解析：{host}（{exc}）") from exc

        for info in infos:
            address = info[4][0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            self._reject_unsafe(ip, host)

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
        # 198.18.0.0/15 单独说明：它几乎总是本机代理软件 fake-ip 模式的占位地址，
        # 而不是真的内网服务，直接提示用户加白名单比让他放开整个内网更安全。
        hint = (
            f"（{ip} 属于 fake-ip 常用段，若你本机开着 Clash/Surge 一类代理的 fake-ip 模式，"
            f"把 {host} 加进配置项 http_allowed_hosts 即可，不必放开整个内网）"
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

        truncated = len(text) > _MAX_BODY_CHARS
        if truncated:
            notice = f"\n……（响应体过长已截断，共 {len(response.text)} 字符）"
            text = text[:_MAX_BODY_CHARS] + notice

        return (
            f"HTTP {response.status_code} {response.reason}\n"
            f"Content-Type: {content_type or '(未声明)'}\n"
            f"最终 URL: {response.url}\n"
            f"{'-' * 40}\n{text}"
        )


def build_tools(config: Any = None) -> list[BaseTool]:
    allow_private = bool(getattr(config, "http_allow_private", False))
    timeout = float(getattr(config, "timeout", 15.0))
    allowed_hosts = list(getattr(config, "http_allowed_hosts", None) or [])
    return [
        HttpRequestTool(allow_private=allow_private, timeout=timeout, allowed_hosts=allowed_hosts)
    ]
