"""核心层：ReAct 主循环与编排。

对应设计方案 §三.4、§四。这里是整个项目唯一同时用到记忆层、工具层、模型层的模块。

主循环（§三.4）::

    把「记忆 + 工具清单」交给模型
       ├─ 有工具调用 → 逐个执行 → 结果回填记忆 → 回到循环顶部
       └─ 无工具调用 → 得到最终答案 → 返回用户，结束

三个兜底必须都在（§八）：
- **最大步数**：防止模型反复调同一工具 / 工具持续报错导致死循环
- **工具异常**：单个工具失败不影响整体，错误信息作为结果回填，模型自己会调整
- **重复调用**：同一工具+同一参数反复调用时跳过执行，直接把上次结果再喂回去
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import BaseLLM, LLMError, LLMResponse, ToolCall
from .memory import Memory
from .prompt import (
    EMPTY_REPLY_NOTICE,
    LOOP_GUARD_NOTICE,
    MAX_STEPS_NOTICE,
    REFUSAL_NOTICE,
    SKIPPED_TOOL_NOTICE,
    TRUNCATED_NOTICE,
)
from .tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """一步的完整记录，用于日志、调试和 CLI 展示（§三.4 可观测性）。"""

    index: int
    kind: str  # tool | final | error | max_steps
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    elapsed: float = 0.0
    text: str = ""  # 模型这一步说的话


@dataclass
class AgentResult:
    """一次 run() 的结果。"""

    answer: str
    steps: int = 0
    stop_reason: str = "final"  # final | max_steps | error | refusal | cancelled
    trace: list[StepRecord] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    #: 这一轮里模型调用的耗时统计（秒）。和 ``usage`` 一样是跨步累加的：
    #:
    #: - ``llm_seconds``     所有模型调用从发出到收完的总耗时
    #: - ``decode_seconds``  其中的解码窗口总和（去掉首字等待的那部分）
    #:
    #: 两个都留着是因为它们的用途不同：算 token 速度要用 ``decode_seconds``
    #: （首字等待里模型一个 token 都没吐，算进去速度会低一大截），
    #: 而 ``llm_seconds`` 才是「这一轮在模型上花了多久」的真实答案。
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("final", "refusal")


class Agent:
    """ReAct Agent。依赖全部从外面注入，方便替换和测试。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        memory: Memory,
        config: Any = None,
        on_step: Callable[[StepRecord], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.config = config
        self.on_step = on_step
        self.on_text = on_text

        self.max_steps = int(getattr(config, "max_steps", 8) or 8)
        self.stream = bool(getattr(config, "stream", True))
        self.max_identical_calls = int(getattr(config, "max_identical_calls", 3) or 3)

        #: 本轮里「同一工具 + 同一参数」的出现次数，用于打断原地打转
        self._call_counts: dict[str, int] = {}
        #: 同一调用的上一次结果，重复时直接复用
        self._call_cache: dict[str, str] = {}
        #: 中断信号（§三.4「可打断」）。界面层在另一个线程里 set，主循环在检查点响应。
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    # 中断
    # ------------------------------------------------------------------ #

    def cancel(self) -> None:
        """请求中断当前这一轮（线程安全，随时可调）。

        三个检查点响应它：每步开头、模型返回后、执行每个工具**之前**。
        之所以要等检查点而不是立刻抛异常，是为了不让中断横穿模型层——
        代价是「正在生成的那一次调用」会跑完（用户那边立刻停止上屏），
        但绝不会有工具在用户喊停之后还被执行。
        """
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> AgentResult:
        """跑完一轮 ReAct 循环，返回最终答案。"""
        self.memory.add_user(user_input)
        self._call_counts.clear()
        self._call_cache.clear()
        self._cancel.clear()

        trace: list[StepRecord] = []
        usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        timing: dict[str, float] = {"llm_seconds": 0.0, "decode_seconds": 0.0}

        for step in range(1, self.max_steps + 1):
            if self._cancel.is_set():  # 检查点 1：用户在上一步之后喊停
                return self._cancelled_result(trace, usage, timing, step - 1)
            logger.debug("=== 第 %d/%d 步 ===", step, self.max_steps)
            messages = self.memory.get_messages()

            try:
                response = self._call_llm(messages)
            except LLMError as exc:
                logger.error("模型调用失败：%s", exc)
                record = StepRecord(index=step, kind="error", text=str(exc))
                trace.append(record)
                self._emit_step(record)
                return self._result(f"模型调用失败：{exc}", "error", trace, usage, timing, step)

            self._accumulate_usage(usage, response)
            self._accumulate_timing(timing, response)

            # 检查点 2：模型是在用户喊停之后才返回的（边流边看时最常见）。
            # 这一次调用的 token 已经花了，但结果作废，也不进记忆。
            if self._cancel.is_set():
                return self._cancelled_result(trace, usage, timing, step)

            # ---- 没有工具调用：得到最终答案，结束 ----
            if not response.has_tool_calls:
                answer = self._final_answer(response)
                self.memory.add_assistant(
                    response.content, [], raw=response.raw, provider=self.llm.provider
                )
                record = StepRecord(index=step, kind="final", text=answer)
                trace.append(record)
                self._emit_step(record)
                logger.debug("循环结束：第 %d 步拿到最终答案", step)
                return self._result(
                    answer,
                    "refusal" if response.stop_reason == "refusal" else "final",
                    trace,
                    usage,
                    timing,
                    step,
                )

            # ---- 有工具调用：先回填 assistant 消息，再逐个执行 ----
            # 这一步很关键：带 tool_calls 的 assistant 消息和后面的 tool 消息必须成对进记忆
            self.memory.add_assistant(
                response.content,
                response.tool_calls,
                raw=response.raw,
                provider=self.llm.provider,
            )

            if response.content.strip():
                logger.info("模型：%s", _short(response.content))

            for position, call in enumerate(response.tool_calls):
                # 检查点 3：**执行之前**。工具可能写文件、发请求，喊停之后绝不能再动。
                if self._cancel.is_set():
                    # 但助手消息已经带着这一批 tool_calls 进记忆了，剩下的必须补上占位结果，
                    # 否则就留下「有 tool_calls 没有 tool 结果」的非法配对，下一轮直接 400。
                    self._backfill_skipped(response.tool_calls[position:])
                    return self._cancelled_result(trace, usage, timing, step)
                record = self._execute_call(step, call)
                trace.append(record)
                self._emit_step(record)

        # ---- 达到最大步数：兜底退出 ----
        logger.warning("达到最大步数 %d，强制结束", self.max_steps)
        answer = self._max_steps_answer(trace)
        self.memory.add_assistant(answer, [], raw=None, provider=self.llm.provider)
        record = StepRecord(index=self.max_steps, kind="max_steps", text=answer)
        trace.append(record)
        self._emit_step(record)
        return self._result(answer, "max_steps", trace, usage, timing, self.max_steps)

    # ------------------------------------------------------------------ #
    # 内部：模型调用
    # ------------------------------------------------------------------ #

    def _call_llm(self, messages: list[dict[str, Any]]) -> LLMResponse:
        schemas = self.tools.schemas() if len(self.tools) else None
        logger.debug("交给模型：%d 条消息，%d 个工具", len(messages), len(schemas or []))

        if self.stream and self.llm.supports_streaming:
            return self.llm.stream_chat(messages, schemas, on_text=self._stream_callback)
        return self.llm.chat(messages, schemas)

    def _stream_callback(self, chunk: str) -> None:
        """转交给上层的 on_text，但用户喊停之后就不再往外吐字（界面上立刻停住）。"""
        if self._cancel.is_set():
            return
        if self.on_text:
            self.on_text(chunk)

    def _backfill_skipped(self, calls: list[ToolCall]) -> None:
        """给「因为中断而没执行」的工具调用补占位结果，保住消息配对。

        为什么要补：带 tool_calls 的 assistant 消息一旦进了记忆，后面就必须跟同样数量的
        tool 消息。用户在多工具的一批中间喊停时，剩下的调用没有结果——不补的话这一轮
        看着没事，**下一轮**请求会因为消息序列非法被服务端拒掉（400），而且报错信息
        通常指不到真正的原因上。

        占位结果标记为错误：工具确实没跑成功，模型下一轮看到它就该知道这一步是空的。
        """
        for call in calls:
            logger.debug("中断：跳过工具 %s(%s)", call.name, _short(str(call.arguments)))
            self.memory.add_tool_result(call.id, call.name, SKIPPED_TOOL_NOTICE, is_error=True)

    def _cancelled_result(
        self,
        trace: list[StepRecord],
        usage: dict[str, int],
        timing: dict[str, float],
        step: int,
    ) -> AgentResult:
        logger.info("本轮对话被用户中断（第 %d 步）", step)
        return self._result("", "cancelled", trace, usage, timing, step)

    @staticmethod
    def _result(
        answer: str,
        stop_reason: str,
        trace: list[StepRecord],
        usage: dict[str, int],
        timing: dict[str, float],
        step: int,
    ) -> AgentResult:
        """所有出口都从这里走。

        五个 return 分支各拼一遍 AgentResult 的话，「新增一个统计字段」就要改五处、
        还必然会漏一处——统计口径恰恰是最不该存在多个版本的东西。
        """
        return AgentResult(
            answer=answer,
            steps=step,
            stop_reason=stop_reason,
            trace=trace,
            usage=usage,
            timing=timing,
        )

    @staticmethod
    def _accumulate_usage(total: dict[str, int], response: LLMResponse) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            value = response.usage.get(key) if isinstance(response.usage, dict) else None
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value

    @staticmethod
    def _accumulate_timing(total: dict[str, float], response: LLMResponse) -> None:
        """把这一步的模型耗时并进这一轮的总账（跨步累加，和 token 数一个口径）。"""
        total["llm_seconds"] = total.get("llm_seconds", 0.0) + float(response.elapsed or 0.0)
        total["decode_seconds"] = total.get("decode_seconds", 0.0) + float(
            response.decode_seconds or 0.0
        )

    def _final_answer(self, response: LLMResponse) -> str:
        if response.stop_reason == "refusal":
            logger.warning("模型触发安全拒绝")
            return REFUSAL_NOTICE
        answer = response.content.strip()
        if not answer:
            logger.warning("模型既没有返回内容也没有工具调用")
            return EMPTY_REPLY_NOTICE
        if response.stop_reason == "max_tokens":
            logger.warning("输出被 max_tokens 截断")
            answer += TRUNCATED_NOTICE
        return answer

    # ------------------------------------------------------------------ #
    # 内部：工具执行
    # ------------------------------------------------------------------ #

    def _execute_call(self, step: int, call: ToolCall) -> StepRecord:
        signature = self._signature(call)
        self._call_counts[signature] = self._call_counts.get(signature, 0) + 1
        times = self._call_counts[signature]

        # 同一个调用原地打转：跳过执行，把上次结果再喂回去（§十「模型反复调同一工具」）
        if self.max_identical_calls and times > self.max_identical_calls:
            previous = self._call_cache.get(signature, "（上次没有拿到结果）")
            logger.warning("工具 %s 用相同参数被调用第 %d 次，已跳过", call.name, times)
            content = LOOP_GUARD_NOTICE.format(times=times, last_result=_short(previous, 500))
            self.memory.add_tool_result(call.id, call.name, content, is_error=True)
            return StepRecord(
                index=step,
                kind="tool",
                tool_name=call.name,
                arguments=call.arguments,
                result=content,
                is_error=True,
            )

        # 参数解析失败：不执行，直接把错误交回模型让它自己修
        if call.parse_error:
            content = (
                f"工具 {call.name} 的参数不是合法 JSON（{call.parse_error}）。"
                f"你传的原文是：{call.raw_arguments[:300]}\n请重新调用并传入合法 JSON。"
            )
            self.memory.add_tool_result(call.id, call.name, content, is_error=True)
            return StepRecord(
                index=step,
                kind="tool",
                tool_name=call.name,
                arguments={},
                result=content,
                is_error=True,
            )

        args_preview = _short(json.dumps(call.arguments, ensure_ascii=False))
        logger.info("调用工具 %s(%s)", call.name, args_preview)
        started = time.perf_counter()
        result: ToolResult = self.tools.execute(call.name, call.arguments)
        elapsed = time.perf_counter() - started

        self._call_cache[signature] = result.content
        self.memory.add_tool_result(call.id, call.name, result.content, result.is_error)

        if result.is_error:
            logger.warning("工具 %s 返回错误：%s", call.name, _short(result.content))
        else:
            logger.info("工具 %s 返回 %d 字符，耗时 %.2fs", call.name, len(result.content), elapsed)

        return StepRecord(
            index=step,
            kind="tool",
            tool_name=call.name,
            arguments=call.arguments,
            result=result.content,
            is_error=result.is_error,
            elapsed=elapsed,
        )

    @staticmethod
    def _signature(call: ToolCall) -> str:
        try:
            arguments = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = str(call.arguments)
        return f"{call.name}::{arguments}"

    def _max_steps_answer(self, trace: list[StepRecord]) -> str:
        results = [record for record in trace if record.kind == "tool" and not record.is_error]
        if results:
            summary = "\n".join(
                f"- {record.tool_name}：{_short(record.result, 200)}" for record in results[-5:]
            )
        else:
            summary = "- （没有成功执行任何工具）"
        return MAX_STEPS_NOTICE.format(steps=self.max_steps, summary=summary)

    # ------------------------------------------------------------------ #
    # 杂项
    # ------------------------------------------------------------------ #

    def _emit_step(self, record: StepRecord) -> None:
        if self.on_step:
            try:
                self.on_step(record)
            except Exception:  # 展示回调出错不该影响主循环
                logger.exception("on_step 回调出错")

    def reset(self) -> None:
        """清空对话历史（保留 system 提示）。"""
        self.memory.clear()
        self._call_counts.clear()
        self._call_cache.clear()


def _short(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "……"
