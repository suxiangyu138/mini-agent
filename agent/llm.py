"""模型层：统一接口 + 各厂商适配器。

对应设计方案 §三.1。核心约定：

- **上层只认识一种消息格式**（见下面的「统一消息格式」），厂商差异全部在这一层吃掉。
- 统一输入：消息列表 + 工具描述列表（可选）
- 统一输出：:class:`LLMResponse` ``{content, tool_calls}``

统一消息格式（内部规范形，四个角色严格区分，§三.3）::

    {"role": "system",    "content": "..."}
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "content": "...", "tool_call_id": "call_xxx", "is_error": False}

想接一个新厂商：继承 :class:`BaseLLM`，实现 ``_convert_messages`` / ``_convert_tools`` /
``chat``（流式可选），然后登记进 :data:`PROVIDER_PRESETS`。上层一行都不用改。
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: 适配器把厂商原始返回挂在这个键上，用于同厂商多轮续接（如 Anthropic 的 thinking 块）。
#: 换个厂商就会忽略它并从句柄重建，所以不会污染统一格式。
_RAW_KEY = "_provider_raw"

OnText = Callable[[str], None]


class LLMError(Exception):
    """模型层统一异常。适配器把各家的异常翻译成它，核心层只需处理这一种。"""


# --------------------------------------------------------------------------- #
# 统一数据结构
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    """一次工具调用请求。``arguments`` 一定是 dict（解析失败时为空 dict）。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            # 个别厂商（或流式分片）不给 id，自己补一个，保证 tool_result 能对上号
            self.id = f"call_{abs(hash((self.name, self.raw_arguments))) % 10**8}"


@dataclass
class LLMResponse:
    """模型返回的统一结构。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    #: 这次调用从发出到收完的总耗时（秒）。
    elapsed: float = 0.0
    #: **真正的解码窗口**：第一个增量到最后一个增量之间（秒）。
    #:
    #: 和 ``elapsed`` 的差值就是首字等待（排队 + prefill），那段时间模型一个 token 都没吐。
    #: 算 token 速度必须用这个当分母，用 ``elapsed`` 会把速度算低一大截。
    #: 只有流式才测得出来；非流式只有一个整包，无从区分，留 0。
    decode_seconds: float = 0.0
    #: 厂商原始返回，只给同厂商的适配器复用（保住 thinking 块等私有结构）
    raw: Any = None
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def text_or_placeholder(self) -> str:
        return self.content.strip()


# --------------------------------------------------------------------------- #
# 统一消息构造器（核心层/记忆层只用这三个函数造消息）
# --------------------------------------------------------------------------- #


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_message(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    raw: Any = None,
    provider: str = "",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
        "tool_calls": list(tool_calls or []),
    }
    if raw is not None:
        message[_RAW_KEY] = {"provider": provider, "content": raw}
    return message


def tool_message(
    tool_call_id: str, name: str, content: str, is_error: bool = False
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
        "is_error": bool(is_error),
    }


def system_message(text: str) -> dict[str, Any]:
    return {"role": "system", "content": text}


def normalize_usage(raw: Any) -> dict[str, Any]:
    """把各家的 token 统计统一成一套键名（``input_tokens`` / ``output_tokens``）。

    OpenAI 兼容这一家子叫 ``prompt_tokens`` / ``completion_tokens``，Anthropic 叫
    ``input_tokens`` / ``output_tokens``。核心层和入口层只认后者，换厂商时统计口径不跟着变。
    原始键名保留不动，想看的还能看到。
    """
    if not isinstance(raw, dict):
        return {}
    usage = dict(raw)
    if "input_tokens" not in usage and isinstance(usage.get("prompt_tokens"), int):
        usage["input_tokens"] = usage["prompt_tokens"]
    if "output_tokens" not in usage and isinstance(usage.get("completion_tokens"), int):
        usage["output_tokens"] = usage["completion_tokens"]
    details = usage.get("prompt_tokens_details")
    if "cache_read_input_tokens" not in usage and isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            usage["cache_read_input_tokens"] = cached
    return usage


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #


class BaseLLM(ABC):
    """模型层统一接口。"""

    provider: str = "base"
    supports_streaming: bool = False

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.model = str(getattr(config, "model", "") or "")
        self.max_tokens = int(getattr(config, "max_tokens", 16000) or 16000)
        self.timeout = float(getattr(config, "timeout", 60.0) or 60.0)
        temperature = getattr(config, "temperature", None)
        self.temperature = float(temperature) if temperature is not None else None

    @abstractmethod
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """一次性返回完整结果。"""

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: OnText | None = None,
    ) -> LLMResponse:
        """流式返回。

        默认实现退化成「一次性拿完再整段吐出来」——
        这样核心层可以无脑调用，适配器不支持流式也不会报错（§九 第二阶段再逐个补齐）。
        """
        response = self.chat(messages, tools)
        if on_text and response.content:
            on_text(response.content)
        return response

    def list_models(self) -> list[str]:
        """列出这家账号当前可用的模型名。

        模型换代很快（预设里写的只是「当时的旗舰」），所以留了这么一个口子：
        想知道有哪些新模型，直接问厂商，别猜。返回空列表表示这家问不出来。
        """
        return []

    # ---------- 给子类用的小工具 ----------

    @staticmethod
    def _parse_arguments(raw: Any, call_id: str, name: str) -> ToolCall:
        """把厂商给的参数（可能是 dict，也可能是 JSON 字符串）统一成 ToolCall。"""
        if isinstance(raw, dict):
            return ToolCall(id=call_id, name=name, arguments=raw)
        text = (raw or "").strip() if isinstance(raw, str) else ""
        if not text:
            return ToolCall(id=call_id, name=name, arguments={})
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # 参数不是合法 JSON：不抛异常，交给注册中心回一句「参数不合法」让模型自己修
            logger.warning("工具 %s 的参数不是合法 JSON：%s（原文：%s）", name, exc, text[:200])
            return ToolCall(
                id=call_id, name=name, arguments={}, raw_arguments=text, parse_error=str(exc)
            )
        if not isinstance(parsed, dict):
            return ToolCall(
                id=call_id,
                name=name,
                arguments={},
                raw_arguments=text,
                parse_error=f"参数应为 JSON 对象，实为 {type(parsed).__name__}",
            )
        return ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=text)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} provider={self.provider} model={self.model}>"


# --------------------------------------------------------------------------- #
# Anthropic 适配器
# --------------------------------------------------------------------------- #


class AnthropicLLM(BaseLLM):
    """Claude 适配器（Messages API）。

    差异点都在这里吃掉：
    - system 是**顶层参数**，不在 messages 里
    - 工具结果要包成 user 消息里的 ``tool_result`` 块，且连续多个结果必须合成**一条** user 消息
    - 工具参数是 dict（不是 JSON 字符串），工具声明用 ``input_schema``
    - 自适应思考默认开启，thinking 块必须原样回传，否则多轮推理质量会掉
    """

    provider = "anthropic"
    supports_streaming = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("未安装 anthropic SDK，请先执行：pip install anthropic") from exc

        self._anthropic = anthropic
        self.enable_thinking = bool(getattr(config, "enable_thinking", True))
        self.show_thinking = bool(getattr(config, "show_thinking", False))
        self.enable_cache = bool(getattr(config, "enable_cache", True))
        self.effort = str(getattr(config, "effort", "") or "")

        api_key = str(getattr(config, "api_key", "") or "") or None
        kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": 2}
        if api_key:
            kwargs["api_key"] = api_key
        base_url = str(getattr(config, "base_url", "") or "")
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self.client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # SDK 在缺 key 时构造就抛
            raise LLMError(
                f"初始化 Anthropic 客户端失败：{exc}\n"
                "请设置环境变量 ANTHROPIC_API_KEY（或 config.json 里的 api_key）。"
            ) from exc

    # ---------- 请求 ----------

    def chat(self, messages, tools=None):
        kwargs = self._build_kwargs(messages, tools)
        started = time.perf_counter()
        message, _ = self._request(kwargs, stream=False)
        response = self._parse_response(message)
        response.elapsed = time.perf_counter() - started
        return response

    def stream_chat(self, messages, tools=None, on_text=None):
        kwargs = self._build_kwargs(messages, tools)
        started = time.perf_counter()
        message, decode_seconds = self._request(kwargs, stream=True, on_text=on_text)
        response = self._parse_response(message)
        response.elapsed = time.perf_counter() - started
        response.decode_seconds = decode_seconds
        return response

    def list_models(self) -> list[str]:
        """Anthropic 的 Models API（SDK 的 client.models.list()）。"""
        try:
            page = self.client.models.list(limit=100)
        except Exception as exc:
            raise LLMError(self._translate(exc)) from exc
        return sorted(item.id for item in page.data if getattr(item, "id", ""))

    def _build_kwargs(self, messages, tools) -> dict[str, Any]:
        system, converted = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            if self.enable_cache:
                # 缓存断点放在 system 块末尾：渲染顺序是 tools → system → messages，
                # 所以工具清单 + 系统提示这一整段稳定前缀都会被缓存
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if self.enable_thinking:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if self.show_thinking:
                thinking["display"] = "summarized"
            kwargs["thinking"] = thinking
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def _request(self, kwargs: dict[str, Any], stream: bool, on_text: OnText | None = None):
        """发请求，返回 ``(原始返回, 解码窗口秒数)``。

        如果模型不支持 thinking / effort，摘掉不支持的参数后重试一次——
        老模型、第三方兼容端点常常不认这两个，与其让用户自己猜，不如按报错降级。
        重试最多一轮，且只在第一次请求失败时发生，不会掩盖真正的错误。
        """
        attempt = dict(kwargs)
        for round_index in range(2):
            try:
                if stream:
                    return self._run_stream(attempt, on_text)
                return self.client.messages.create(**attempt), 0.0
            except Exception as exc:  # 这里统一翻译成 LLMError
                if round_index == 0:
                    reduced = self._strip_unsupported(attempt, str(exc))
                    if reduced is not None:
                        logger.warning("模型 %s 不支持部分参数，已降级重试：%s", self.model, exc)
                        attempt = reduced
                        continue
                raise self._translate(exc) from exc
        raise LLMError("请求失败：重试后仍未成功")  # pragma: no cover

    def _run_stream(self, kwargs: dict[str, Any], on_text: OnText | None) -> tuple[Any, float]:
        first_at: float | None = None
        last_at: float | None = None
        with self.client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                now = time.perf_counter()
                if first_at is None:
                    first_at = now
                last_at = now
                if on_text:
                    on_text(chunk)
            message = stream.get_final_message()
        decode = (last_at - first_at) if first_at and last_at else 0.0
        return message, decode

    def _strip_unsupported(self, kwargs: dict[str, Any], message: str) -> dict[str, Any] | None:
        """按报错内容去掉模型不支持的参数（老模型 / 第三方端点常见）。"""
        reduced = dict(kwargs)
        changed = False
        if "thinking" in reduced and "thinking" in message:
            reduced.pop("thinking")
            changed = True
        if "output_config" in reduced and ("effort" in message or "output_config" in message):
            reduced.pop("output_config")
            changed = True
        system = reduced.get("system")
        if isinstance(system, list) and "cache_control" in message:
            reduced["system"] = "\n\n".join(block.get("text", "") for block in system)
            changed = True
        return reduced if changed else None

    def _translate(self, exc: Exception) -> LLMError:
        anthropic = self._anthropic
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, anthropic.AuthenticationError):
            return LLMError(
                "Anthropic 鉴权失败：API Key 无效或未设置（环境变量 ANTHROPIC_API_KEY）"
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return LLMError("Anthropic 拒绝访问：当前 API Key 没有该模型的权限")
        if isinstance(exc, anthropic.NotFoundError):
            return LLMError(f"模型或接口不存在：{self.model}。请检查 MINI_AGENT_MODEL 配置")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError("触发 Anthropic 限流（429），请稍后重试或降低调用频率")
        if isinstance(exc, anthropic.APIStatusError):
            detail = getattr(exc, "message", "") or str(exc)
            return LLMError(f"Anthropic 返回错误 {getattr(exc, 'status_code', '?')}：{detail}")
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(f"请求超时（{self.timeout:.0f} 秒），可调大 config 里的 timeout")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(f"无法连接 Anthropic（{exc}）。请检查网络或 base_url 配置")
        return LLMError(f"调用 Anthropic 失败：{type(exc).__name__}: {exc}")

    # ---------- 格式转换 ----------

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """统一格式 → Anthropic 格式。返回 (system, messages)。"""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            # 连续多个 tool_result 必须合并成一条 user 消息，否则并行调用会被拆散
            if pending_results:
                converted.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content") or ""))
                continue
            if role == "tool":
                pending_results.append(self._tool_result_block(message))
                continue

            flush_results()
            if role == "user":
                converted.append({"role": "user", "content": str(message.get("content") or "")})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": self._assistant_blocks(message)})
            else:
                logger.warning("忽略无法识别的消息角色：%r", role)

        flush_results()
        return "\n\n".join(part for part in system_parts if part), converted

    def _assistant_blocks(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        # 同一厂商：原样回传上次的 content 块，保住 thinking 块（丢了会掉推理质量）
        raw = message.get(_RAW_KEY)
        if isinstance(raw, dict) and raw.get("provider") == self.provider and raw.get("content"):
            return list(raw["content"])

        blocks: list[dict[str, Any]] = []
        text = str(message.get("content") or "")
        if text:
            blocks.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments if isinstance(call.arguments, dict) else {},
                }
            )
        return blocks or [{"type": "text", "text": "(无内容)"}]

    @staticmethod
    def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(message.get("tool_call_id") or ""),
            "content": str(message.get("content") or ""),
        }
        if message.get("is_error"):
            block["is_error"] = True
        return block

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> LLMResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        stop_reason = getattr(response, "stop_reason", "") or ""
        if stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            logger.warning("模型拒绝了本次请求：%s", getattr(detail, "category", None))

        usage = getattr(response, "usage", None)
        return LLMResponse(
            content="".join(texts).strip(),
            tool_calls=calls,
            stop_reason=stop_reason,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "cache_read_input_tokens": (
                    getattr(usage, "cache_read_input_tokens", 0) if usage else 0
                ),
            },
            raw=response.content,  # 原样留着，下一轮原封不动传回去
            model=getattr(response, "model", "") or self.model,
        )


# --------------------------------------------------------------------------- #
# OpenAI 兼容适配器（OpenAI / DeepSeek / 通义 / Moonshot / 智谱 / vLLM / Ollama…）
# --------------------------------------------------------------------------- #


class OpenAICompatLLM(BaseLLM):
    """OpenAI ``/chat/completions`` 协议的通用适配器。

    国内主流厂商和本地推理框架几乎都提供兼容端点，
    所以这一个适配器能覆盖一大片（改 base_url 和 model 即可）。
    用 ``requests`` 直接发 HTTP，避免为了「兼容协议」再装一个 SDK。
    """

    provider = "openai"
    supports_streaming = True
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.base_url = (
            str(getattr(config, "base_url", "") or "") or self.default_base_url
        ).rstrip("/")
        self.api_key = str(getattr(config, "api_key", "") or "")

    # ---------- 请求 ----------

    def chat(self, messages, tools=None):
        payload = self._build_payload(messages, tools, stream=False)
        started = time.perf_counter()
        data = self._post(payload, stream=False)
        response = self._parse_response(data)
        response.elapsed = time.perf_counter() - started
        return response

    def stream_chat(self, messages, tools=None, on_text=None):
        payload = self._build_payload(messages, tools, stream=True)
        started = time.perf_counter()
        try:
            response = self._post(payload, stream=True, on_text=on_text)
        except LLMError as exc:
            # 个别兼容端点不认 stream_options，去掉重来一次（少了的只是 token 统计）。
            # 报错发生在收到任何内容之前，所以重试不会重复吐字。
            if "stream_options" not in str(exc) or "stream_options" not in payload:
                raise
            logger.debug("该端点不支持 stream_options，去掉后重试")
            payload.pop("stream_options")
            started = time.perf_counter()  # 重试算一次新的，别把失败那次的等待算进速度
            response = self._post(payload, stream=True, on_text=on_text)
        response.elapsed = time.perf_counter() - started
        return response

    def list_models(self) -> list[str]:
        """GET /models。OpenAI 兼容这一家子基本都实现了，顺手也把 Key 验了。"""
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise LLMError("未安装 requests，请先执行：pip install requests") from exc

        url = f"{self.base_url}/models"
        try:
            response = requests.get(url, headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"无法连接 {url}：{exc}") from exc
        if response.status_code >= 400:
            raise LLMError(self._error_message(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"接口返回的不是 JSON：{response.text[:300]}") from exc
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        ids = [str(item["id"]) for item in entries if isinstance(item, dict) and item.get("id")]
        return sorted(ids)

    def _build_payload(self, messages, tools, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
            # 流式默认不带 token 统计，得主动要（不支持这家的端点会自动去掉重试）
            payload["stream_options"] = {"include_usage": True}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any], stream: bool, on_text: OnText | None = None) -> Any:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise LLMError("未安装 requests，请先执行：pip install requests") from exc

        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
        except requests.Timeout as exc:
            raise LLMError(f"请求超时（{self.timeout:.0f} 秒）：{url}") from exc
        except requests.RequestException as exc:
            raise LLMError(f"无法连接 {url}：{exc}") from exc

        if response.status_code >= 400:
            raise LLMError(self._error_message(response))

        if stream:
            return self._consume_stream(response, on_text)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"接口返回的不是 JSON：{response.text[:300]}") from exc

    def _error_message(self, response: Any) -> str:
        detail = response.text[:300]
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = error.get("message") or detail
                elif isinstance(error, str):
                    detail = error
                else:
                    detail = body.get("message") or detail
        except ValueError:
            pass

        hints = {
            401: "API Key 无效或未设置",
            403: "没有访问权限",
            404: f"接口或模型不存在（{self.model}），请检查 base_url 与 model 配置",
            429: "触发限流，请稍后重试",
        }
        hint = hints.get(response.status_code, "")
        suffix = f"（{hint}）" if hint else ""
        return f"模型接口返回 {response.status_code}{suffix}：{detail}"

    def _consume_stream(self, response: Any, on_text: OnText | None) -> LLMResponse:
        """解析 SSE 流：文本边收边吐，工具调用的分片按 index 拼回去。"""
        texts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        first_at: float | None = None
        last_at: float | None = None

        def arriving() -> None:
            """有一个增量到了。计时只看增量，不看请求发出和首字之前的等待。"""
            nonlocal first_at, last_at
            now = time.perf_counter()
            if first_at is None:
                first_at = now
            last_at = now

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("跳过无法解析的流式分片：%s", data[:120])
                continue

            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                texts.append(piece)
                arriving()
                if on_text:
                    on_text(piece)

            for fragment in delta.get("tool_calls") or []:
                # 工具调用的参数也是模型一个 token 一个 token 吐出来的，同样算解码时间。
                # 不把这段计进去，纯工具调用那一轮的解码窗口就会是 0，
                # 那些 token 会被算成「不花时间」，速度直接虚高几倍。
                arriving()
                index = int(fragment.get("index", 0) or 0)
                slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if fragment.get("id"):
                    slot["id"] = fragment["id"]
                function = fragment.get("function") or {}
                if function.get("name"):
                    slot["name"] += function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

        calls = [
            self._parse_arguments(slot["arguments"], slot["id"], slot["name"])
            for _, slot in sorted(slots.items())
            if slot["name"]
        ]
        return LLMResponse(
            content="".join(texts).strip(),
            tool_calls=calls,
            stop_reason=finish_reason or ("tool_calls" if calls else "stop"),
            usage=normalize_usage(usage),
            decode_seconds=(last_at - first_at) if first_at and last_at else 0.0,
            model=self.model,
        )

    # ---------- 格式转换 ----------

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role in ("system", "user"):
                converted.append({"role": role, "content": str(message.get("content") or "")})
            elif role == "assistant":
                calls = message.get("tool_calls") or []
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": str(message.get("content") or "") or None,
                }
                if calls:
                    item["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                # 参数必须以**字符串**形式回传，这里重新序列化保证是合法 JSON
                                "arguments": json.dumps(call.arguments or {}, ensure_ascii=False),
                            },
                        }
                        for call in calls
                    ]
                converted.append(item)
            elif role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(message.get("tool_call_id") or ""),
                        "content": str(message.get("content") or ""),
                    }
                )
            else:
                logger.warning("忽略无法识别的消息角色：%r", role)
        return converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    def _parse_response(self, data: Any) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            preview = json.dumps(data, ensure_ascii=False)[:300]
            raise LLMError(f"接口没有返回 choices 字段：{preview}")

        message = choices[0].get("message") or {}
        calls = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            calls.append(
                self._parse_arguments(
                    function.get("arguments"), item.get("id", ""), function.get("name", "")
                )
            )

        return LLMResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            stop_reason=choices[0].get("finish_reason") or "",
            usage=normalize_usage(data.get("usage")),
            model=data.get("model", "") or self.model,
        )


class OllamaLLM(OpenAICompatLLM):
    """Ollama 走它的 OpenAI 兼容端点，省得再写一套 /api/chat 的解析。"""

    provider = "ollama"
    default_base_url = "http://localhost:11434/v1"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# --------------------------------------------------------------------------- #
# 离线模拟适配器（没有 API Key 也能跑通整条链路）
# --------------------------------------------------------------------------- #


class MockLLM(BaseLLM):
    """不联网的假模型：用来做单元测试、离线演示主循环。

    两种模式：
    - 传入 ``script``：按顺序返回预设的 :class:`LLMResponse`（测试用，完全确定）
    - 不传：走内置启发式 —— 问题里有算式就先调 calculator，再调 get_current_time，
      最后把工具结果汇总成回答。用于 ``python main.py --demo`` 离线看多步循环。
    """

    provider = "mock"
    supports_streaming = True

    def __init__(self, config: Any = None, script: list[LLMResponse] | None = None) -> None:
        if config is None:
            config = type("_MockConfig", (), {"model": "mock", "max_tokens": 1024})()
        super().__init__(config)
        self.model = self.model or "mock"
        self.script = list(script or [])

    def chat(self, messages, tools=None):
        if self.script:
            response = self.script.pop(0)
            logger.debug("MockLLM 返回脚本第 %d 条", len(self.script))
            return response
        return self._heuristic(messages, tools or [])

    def stream_chat(self, messages, tools=None, on_text=None):
        response = self.chat(messages, tools)
        if on_text and response.content:
            # 按句切开，模拟打字机效果
            for piece in _chunks(response.content, 12):
                on_text(piece)
        return response

    def _heuristic(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        import re

        names = {tool["name"] for tool in tools}
        last_user_index = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"), default=0
        )
        question = str(messages[last_user_index].get("content") or "")
        after = messages[last_user_index:]
        results = [m for m in after if m.get("role") == "tool"]
        done = {m.get("name") for m in results}

        # 第一步：问题里有算式 → 调计算器
        if "calculator" in names and "calculator" not in done:
            expression = _find_expression(question)
            if expression:
                call = ToolCall(
                    id="mock-calc-1",
                    name="calculator",
                    arguments={"expression": expression},
                )
                return LLMResponse(
                    content="我先用计算器算一下。",
                    tool_calls=[call],
                    stop_reason="tool_calls",
                )

        # 第二步：问题问到时间 → 调时间工具
        if "get_current_time" in names and "get_current_time" not in done:
            if re.search(r"今天|现在|当前|日期|时间|几号|星期", question):
                call = ToolCall(
                    id="mock-time-1",
                    name="get_current_time",
                    arguments={"timezone": "local"},
                )
                return LLMResponse(
                    content="再查一下当前时间。",
                    tool_calls=[call],
                    stop_reason="tool_calls",
                )

        # 收尾：把工具结果汇总
        if results:
            summary = "\n".join(f"- {m.get('content', '').strip()}" for m in results)
            return LLMResponse(
                content=f"（离线模拟）根据工具返回：\n{summary}", stop_reason="end_turn"
            )

        return LLMResponse(
            content=(
                "（离线模拟）我收到了你的问题：" + question.strip() + "\n"
                "当前用的是 mock 模型，没有接真实 LLM。"
                "配置 API Key 后换成真实 provider 即可。"
            ),
            stop_reason="end_turn",
        )


def _chunks(text: str, size: int):
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _find_expression(text: str) -> str:
    """从自然语言里抠出一个能算的算式，给 mock 模型用。

    「帮我算一下 (1234 * 5678) / 9 是多少」要能抠出 ``(1234 * 5678) / 9``，
    所以片段允许以 ``(`` 开头，并在括号不配平时就地修正。
    """
    import re

    pattern = re.compile(r"[(\d][\d\s.+\-*/%()]*\d")
    best = ""
    for match in pattern.finditer(text or ""):
        candidate = _balance_brackets(match.group().strip())
        if re.search(r"[+\-*/%]", candidate) and len(candidate) > len(best):
            best = candidate
    return best


def _balance_brackets(candidate: str) -> str:
    """把抠出来的片段修成括号配平的表达式。"""
    depth = 0
    end = len(candidate)
    for index, char in enumerate(candidate):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:  # 多出来一个右括号：从它开始的内容都不属于这个算式
                end = index
                break
    clipped = candidate[:end].strip()
    return clipped + ")" * depth if depth > 0 else clipped


# --------------------------------------------------------------------------- #
# 厂商登记表 + 工厂
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderPreset:
    """一个厂商的接入信息：用哪个适配器、默认端点、从哪个环境变量读 Key、示例模型。"""

    name: str
    adapter: type[BaseLLM]
    env_key: str
    default_model: str
    base_url: str = ""
    note: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "anthropic": ProviderPreset(
        name="anthropic",
        adapter=AnthropicLLM,
        env_key="ANTHROPIC_API_KEY",
        default_model="claude-opus-5",
        note="Anthropic 官方 SDK，支持自适应思考与流式",
    ),
    "openai": ProviderPreset(
        name="openai",
        adapter=OpenAICompatLLM,
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o",
        base_url="https://api.openai.com/v1",
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        adapter=OpenAICompatLLM,
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        adapter=OpenAICompatLLM,
        env_key="DASHSCOPE_API_KEY",
        default_model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        note="通义千问（DashScope OpenAI 兼容模式）",
    ),
    "moonshot": ProviderPreset(
        name="moonshot",
        adapter=OpenAICompatLLM,
        env_key="MOONSHOT_API_KEY",
        default_model="kimi-k3",
        base_url="https://api.moonshot.cn/v1",
        note="月之暗面 Kimi（别名 kimi）",
    ),
    "zhipu": ProviderPreset(
        name="zhipu",
        adapter=OpenAICompatLLM,
        env_key="ZHIPU_API_KEY",
        default_model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        note="智谱 GLM（别名 glm）",
    ),
    "minimax": ProviderPreset(
        name="minimax",
        adapter=OpenAICompatLLM,
        env_key="MINIMAX_API_KEY",
        default_model="MiniMax-M3",
        base_url="https://api.minimaxi.com/v1",
    ),
    "mimo": ProviderPreset(
        name="mimo",
        adapter=OpenAICompatLLM,
        env_key="MIMO_API_KEY",
        default_model="mimo-v2.5-pro",
        base_url="https://api.xiaomimimo.com/v1",
        note="小米 MiMo（别名 xiaomi）",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        adapter=OllamaLLM,
        env_key="",
        default_model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        note="本地模型，无需 API Key（记得先 ollama pull）",
    ),
    "openai_compatible": ProviderPreset(
        name="openai_compatible",
        adapter=OpenAICompatLLM,
        env_key="OPENAI_API_KEY",
        default_model="",
        note="任何兼容 OpenAI 协议的端点：自己填 base_url 和 model",
    ),
    "mock": ProviderPreset(
        name="mock",
        adapter=MockLLM,
        env_key="",
        default_model="mock",
        note="离线模拟，不联网，用于演示和测试",
    ),
}

#: 常见别名 → 规范名
PROVIDER_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "anthropic_claude": "anthropic",
    "gpt": "openai",
    "azure_openai": "openai_compatible",
    "dashscope": "qwen",
    "tongyi": "qwen",
    "通义": "qwen",
    "kimi": "moonshot",
    "glm": "zhipu",
    "bigmodel": "zhipu",
    "xiaomi": "mimo",
    "mimo_ai": "mimo",
    "local": "ollama",
    "compatible": "openai_compatible",
    "fake": "mock",
    "dummy": "mock",
    "echo": "mock",
}


def resolve_provider(name: str) -> str:
    """把别名解析成规范 provider 名。"""
    key = (name or "").strip().lower()
    key = PROVIDER_ALIASES.get(key, key)
    if key not in PROVIDER_PRESETS:
        available = ", ".join(sorted(PROVIDER_PRESETS))
        raise LLMError(f"未知的 provider：{name!r}。可选：{available}")
    return key


def create_llm(config: Any) -> BaseLLM:
    """工厂：**换模型后端只改这一处**（§十一 验收标准 2）。

    上层（core/main）只依赖 BaseLLM，永远不 import 具体适配器。
    """
    provider = resolve_provider(getattr(config, "provider", "anthropic"))
    preset = PROVIDER_PRESETS[provider]
    llm = preset.adapter(config)
    # 一个适配器服务多个厂商（OpenAI 兼容协议那一大家子），把实例上的 provider
    # 标成用户实际选的那个，日志和界面才不会显示成 openai。
    llm.provider = provider
    logger.debug("已创建模型适配器：%s（model=%s）", provider, llm.model)
    return llm
