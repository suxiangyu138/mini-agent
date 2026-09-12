"""一轮对话的计时与指标口径（用户明确要求界面上要有输入/输出 token 和 token 速度）。

单独一个模块是为了打断环形依赖：:mod:`web.sessions` 要在收尾时算出指标，
:mod:`web.server` 要把指标写进 SSE——两边都需要它，而它们不能互相 import。
放在这里之后依赖是单向的：``server → sessions → telemetry → agent``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent import AgentResult, StepRecord

#: stop_reason → 给界面看的中文说法
STOP_REASONS = {
    "final": "完成",
    "refusal": "模型拒绝回答",
    "max_steps": "达到步数上限",
    "cancelled": "已中断",
    "error": "出错",
}


@dataclass
class TurnClock:
    """给一轮对话计时。

    要算「token 速度」就必须把**生成窗口**和**整轮耗时**分开：
    整轮里混着工具执行、网络往返、排队，拿它当分母会把速度算得莫名其妙地低。
    生成窗口 = 第一个字到最后一个字之间的那段，这才是模型真正在解码的时间。
    """

    started: float
    first_chunk: float | None = None
    last_chunk: float | None = None
    chars: int = 0

    def mark(self, chunk: str) -> None:
        now = time.perf_counter()
        if self.first_chunk is None:
            self.first_chunk = now
        self.last_chunk = now
        self.chars += len(chunk)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def ttft(self) -> float:
        """首字延迟：从提问到看见第一个字。流式体验里最敏感的一个数。"""
        if self.first_chunk is None:
            return 0.0
        return self.first_chunk - self.started

    def generation_window(self) -> float:
        """生成窗口（秒）。没走流式时退化成整轮耗时，否则速度无从谈起。

        判断「有没有收到过分片」必须拿 ``is None`` 比，不能图省事写 ``if self.first_chunk``
        ——时间戳是个 float，0.0 是合法值，（测试里就撞上了）会被当成「没收到」。
        """
        if self.first_chunk is not None and self.last_chunk is not None:
            window = self.last_chunk - self.first_chunk
            if window > 0.05:
                return window
        return max(self.elapsed, 1e-6)


def step_payload(record: StepRecord) -> dict[str, Any]:
    """把 StepRecord 翻成前端要的形状（可观测性，§三.4）。"""
    return {
        "index": record.index,
        "kind": record.kind,
        "tool": record.tool_name,
        "arguments": record.arguments,
        "result": record.result,
        "is_error": record.is_error,
        "elapsed": round(record.elapsed, 3),
        "text": record.text,
    }


#: 解码窗口短于这个秒数就不报「解码速率」。见 metrics_payload 的说明。
_MIN_TRUSTED_WINDOW = 0.5


def metrics_payload(result: AgentResult, clock: TurnClock) -> dict[str, Any]:
    """界面上要显示的那几个数（用户明确要求的：输入/输出 token、token 速度…）。

    **token 速度用「总输出 token ÷ 模型总耗时」，不用解码窗口。** 这是踩过两次坑
    之后才定下来的，理由值得写清楚：

    1. 第一版拿入口层「第一个字到最后一个字」当分母。一轮带工具调用的对话跑出
       460 tok/s —— 分子（``output_tokens``）是**所有**模型调用的总和，分母却只覆盖
       最后那次调用里**看得见的那点文字**，发工具调用那一轮的 token 花了时间没进分母。
    2. 第二版改用模型层测的「解码窗口」（第一个增量到最后一个增量）。数字好看了，
       但立刻发现**厂商并不总是逐字吐**：实测 DeepSeek 一次工具调用把几十个 token
       攒在一个 45ms 的批次里发出来，于是分母 0.045s、速度 600+ tok/s —— 纯属噪声。
       文本回答倒是逐字来的，所以这个坑只在短输出上出现，最难发现。

    结论：分母用 ``llm_seconds``（模型调用从发出到收完的**总**时间，含首字等待）。
    它把排队和 prefill 也算进去了，所以比瞬时解码速度低一些，但它**稳定、可复现、
    不会因为厂商怎么切分片而变**——显示给用户的数，可信比好看重要。

    ``decode_tok_per_s`` 仍然照实给出来（窗口够长时它才准，够短时置 0 表示不可信），
    放在提示气泡里，想深究的人能看到。
    """
    usage = result.usage or {}
    timing = result.timing or {}
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    llm_seconds = float(timing.get("llm_seconds") or 0.0)
    window_source = "模型层"
    if llm_seconds <= 0.05:
        # 适配器没测（非流式），退回入口层自己量的那一段
        llm_seconds = clock.generation_window()
        window_source = "入口层"

    decode_seconds = float(timing.get("decode_seconds") or 0.0)
    decode_speed = 0.0
    if output_tokens and decode_seconds >= _MIN_TRUSTED_WINDOW:
        decode_speed = round(output_tokens / decode_seconds, 1)

    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "tok_per_s": round(output_tokens / llm_seconds, 1) if output_tokens else 0.0,
        "chars": clock.chars,
        "chars_per_s": round(clock.chars / llm_seconds, 1) if clock.chars else 0.0,
        "elapsed": round(clock.elapsed, 2),
        "ttft": round(clock.ttft, 2),
        "llm_seconds": round(llm_seconds, 2),
        "decode_seconds": round(decode_seconds, 2),
        "decode_tok_per_s": decode_speed,
        "window_source": window_source,
        "steps": result.steps,
        "tool_calls": sum(1 for record in result.trace if record.kind == "tool"),
        "stop_reason": result.stop_reason,
        "stop_reason_text": STOP_REASONS.get(result.stop_reason, result.stop_reason),
    }
