"""工具基类与注册中心。

对应设计方案 §三.2 / §八：

- 工具三要素：名字 / 描述 / 参数 Schema，缺一不可
- **工具永远返回字符串**：异常在这一层被兜住，绝不能打爆主循环
- 描述是给模型看的说明书，决定模型会不会用、用得对不对
- 参数在校验后再交给工具，模型多传/少传参数不会直接 TypeError
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


class ToolError(Exception):
    """工具**可预期**的失败：参数不合法、文件不存在、接口超时……

    工具内部抛出它，注册中心会转成一段给模型看的错误说明（``is_error=True``）。
    它既不会打爆主循环，也不会在日志里留下无意义的堆栈。
    真正未预期的 bug 直接让其它异常冒出去，注册中心会记 warning 并带上堆栈。
    """


@dataclass
class ToolResult:
    """工具执行结果。content 一定是字符串，可以直接回填进记忆。"""

    content: str
    is_error: bool = False
    truncated: bool = False


class BaseTool(ABC):
    """所有工具的基类：填三要素 + 实现 run()。

    ``parameters`` 请当作不可变常量使用，不要在 run() 里改它。
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = _EMPTY_SCHEMA

    #: 结果是否受 ToolRegistry 的长度上限约束（``max_result_chars``）。
    #:
    #: **取内容的工具**（http_request、read_file）设 False：正文就是答案本身，
    #: 砍掉一半比没有更糟——模型会拿半截内容当完整内容用，还照样据此下结论。
    #: 这类工具自己交代长度：read_file 有显式的 ``max_bytes`` 参数，
    #: http_request 在头部报出正文字符数。
    truncate_result: bool = True

    def schema(self) -> dict[str, Any]:
        """导出给模型的 JSON Schema（统一格式，适配器负责翻译成各家字段）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or _EMPTY_SCHEMA,
        }

    def validate(self) -> None:
        """注册时的自检：三要素缺一不可，早失败早发现。"""
        if not self.name:
            raise ValueError(f"{type(self).__name__} 缺少 name")
        if not self.description:
            raise ValueError(f"工具 {self.name} 缺少 description —— 描述是给模型看的说明书")
        if not isinstance(self.parameters, dict) or not self.parameters:
            raise ValueError(f"工具 {self.name} 缺少 parameters（JSON Schema）")

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具并返回字符串。可预期的失败请抛 :class:`ToolError`。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name!r}>"


class ToolRegistry:
    """注册中心：按名查找 / 导出全部 Schema / 列出所有名字（设计方案 §三.2）。"""

    def __init__(
        self,
        tools: Iterable[BaseTool] | None = None,
        max_result_chars: int = 8000,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        # 单个工具结果的长度上限：防止一个工具把上下文冲爆（§十「上下文无限增长」）
        self.max_result_chars = max_result_chars
        for tool in tools or []:
            self.register(tool)

    # ---------- 登记 ----------

    def register(self, tool: BaseTool) -> BaseTool:
        tool.validate()
        if tool.name in self._tools:
            logger.warning("工具 %s 被重复注册，后注册的覆盖先前的", tool.name)
        self._tools[tool.name] = tool
        logger.debug("注册工具：%s", tool.name)
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ---------- 查询 ----------

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """导出全部工具的 Schema，交给模型层。"""
        return [tool.schema() for tool in self._tools.values()]

    def describe(self) -> str:
        """人类可读的工具清单，给 CLI 的 --list-tools 和日志用。"""
        lines = []
        for tool in self._tools.values():
            params = ", ".join((tool.parameters.get("properties") or {}).keys()) or "无"
            summary = tool.description.strip().splitlines()[0]
            lines.append(f"  {tool.name}({params})\n    {summary}")
        return "\n".join(lines) if lines else "  （没有注册任何工具）"

    # ---------- 执行 ----------

    def execute(self, name: str, arguments: Any = None) -> ToolResult:
        """执行工具，**永远**返回 ToolResult，绝不抛异常。

        这是「单个工具失败不影响整体」的落点（设计方案 §三.4 / §十）。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(self._unknown_tool_message(name), is_error=True)

        arguments = self._normalize_arguments(arguments)
        if isinstance(arguments, ToolResult):  # 参数解析失败
            return arguments

        try:
            kwargs = self._prepare_arguments(tool, arguments)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)

        try:
            raw = tool.run(**kwargs)
        except ToolError as exc:
            logger.info("工具 %s 报告失败：%s", name, exc)
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:  # 这里就是最后一道兜底
            logger.warning("工具 %s 抛出未预期异常", name, exc_info=True)
            return ToolResult(
                f"工具 {name} 内部错误（{type(exc).__name__}）：{exc}。"
                "请换个思路或换个参数，不要原样重试。",
                is_error=True,
            )

        if raw is None:
            return ToolResult(f"工具 {name} 没有返回内容（工具必须返回字符串）", is_error=True)

        # getattr 而不是直接取属性：这个注册中心收「长得像工具」的类（鸭子类型），
        # 新工具不必继承 BaseTool，也就未必带这个开关。缺省按有上限处理。
        if not getattr(tool, "truncate_result", True):
            return ToolResult(str(raw), is_error=False)
        text, truncated = self._truncate(str(raw))
        return ToolResult(text, is_error=False, truncated=truncated)

    # ---------- 内部工具 ----------

    def _unknown_tool_message(self, name: str) -> str:
        available = ", ".join(self.names()) or "（无）"
        return f"没有名为 {name!r} 的工具。可用工具：{available}。请改用这些工具，或直接回答用户。"

    @staticmethod
    def _normalize_arguments(arguments: Any) -> Any:
        """模型给过来的参数可能是 dict、JSON 字符串、None……先统一成 dict。"""
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                return ToolResult(f"工具参数不是合法 JSON：{exc}", is_error=True)
            if not isinstance(parsed, dict):
                return ToolResult(
                    f"工具参数必须是 JSON 对象，收到 {type(parsed).__name__}", is_error=True
                )
            return parsed
        return ToolResult(
            f"工具参数必须是 JSON 对象，收到 {type(arguments).__name__}", is_error=True
        )

    def _prepare_arguments(self, tool: BaseTool, arguments: dict[str, Any]) -> dict[str, Any]:
        """按 Schema 做轻量校验：丢多余键、做类型纠正、报缺失项。"""
        schema = tool.parameters or _EMPTY_SCHEMA
        properties: dict[str, Any] = schema.get("properties") or {}
        required: Sequence[str] = schema.get("required") or []

        missing = [key for key in required if key not in arguments or arguments[key] is None]
        if missing:
            raise ToolError(
                f"调用 {tool.name} 缺少必填参数：{', '.join(missing)}。"
                f"参数说明：{json.dumps(properties, ensure_ascii=False)}"
            )

        prepared: dict[str, Any] = {}
        dropped: list[str] = []
        for key, value in arguments.items():
            if properties and key not in properties:
                dropped.append(key)
                continue
            prepared[key] = self._coerce(
                value, (properties.get(key) or {}).get("type"), tool.name, key
            )

        if dropped:
            logger.debug("工具 %s 忽略了未声明的参数：%s", tool.name, dropped)
        return prepared

    @staticmethod
    def _coerce(value: Any, expected: str | None, tool_name: str, key: str) -> Any:
        """模型偶尔会把 5 写成 "5"、把 true 写成 "true"，这里做最小纠正。"""
        if expected is None or value is None:
            return value
        try:
            if expected == "integer" and not isinstance(value, bool):
                if isinstance(value, str):
                    return int(float(value.strip()))
                if isinstance(value, float):
                    return int(value)
            elif expected == "number" and not isinstance(value, bool):
                if isinstance(value, str):
                    return float(value.strip())
                if isinstance(value, int):
                    return float(value)
            elif expected == "boolean" and isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "yes", "1"):
                    return True
                if lowered in ("false", "no", "0"):
                    return False
            elif expected == "string" and not isinstance(value, str):
                if isinstance(value, (int, float, bool)):
                    return str(value)
            elif expected == "array" and isinstance(value, str):
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else value
            elif expected == "object" and isinstance(value, str):
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else value
        except (ValueError, json.JSONDecodeError):
            logger.debug("工具 %s 的参数 %s 无法转成 %s，原样透传", tool_name, key, expected)
        return value

    def _truncate(self, text: str) -> tuple[str, bool]:
        limit = self.max_result_chars
        if limit and limit > 0 and len(text) > limit:
            head = text[:limit]
            notice = (
                f"\n\n……（结果过长已截断，共 {len(text)} 字符，仅保留前 {limit} 字符。"
                "如需更精确的内容，请缩小查询范围）"
            )
            return head + notice, True
        return text, False

    # ---------- 语法糖 ----------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())
