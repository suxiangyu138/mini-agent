"""计算器：安全求值，不用 eval。

模型有「能算却硬编」的偷懒倾向（设计方案 §七），所以要给它一个真的会算的工具。
实现方式：``ast.parse`` 解析成语法树后按白名单递归求值，
函数调用、属性访问、下标、推导式等一律拒绝，避免 ``__import__('os').system(...)`` 这类注入。
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from .base import BaseTool, ToolError

Number = int | float

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 白名单里的数学函数与常量。只放纯计算，不放任何能碰 IO 的东西。
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
}

_CONSTANTS: dict[str, Number] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_MAX_NODES = 120  # 表达式规模上限
_MAX_POW_EXPONENT = 1000  # 防止 2 ** 10**9 直接把内存打满
_MAX_MAGNITUDE = 1e300  # 结果量级上限


def _eval_node(node: ast.AST) -> Any:
    """按白名单递归求值，遇到不认识的语法直接拒绝。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"计算器只支持数字常量，不支持 {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ToolError(f"指数 {right} 过大（上限 {_MAX_POW_EXPONENT}），拒绝计算")
        return handler(left, right)

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
        return handler(_eval_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"未知标识符 {node.id!r}。可用常量：{', '.join(sorted(_CONSTANTS))}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolError("只支持直接调用白名单函数，不支持属性调用")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolError(
                f"不支持的函数 {node.func.id!r}。可用函数：{', '.join(sorted(_FUNCTIONS))}"
            )
        if node.keywords:
            raise ToolError("计算器不支持关键字参数")
        return func(*[_eval_node(arg) for arg in node.args])

    raise ToolError(f"表达式里含有不允许的语法：{type(node).__name__}")


def safe_eval(expression: str) -> Number:
    """安全求值一个数学表达式。失败抛 :class:`ToolError`。"""
    text = (expression or "").strip()
    if not text:
        raise ToolError("表达式为空")
    if len(text) > 500:
        raise ToolError("表达式过长（上限 500 字符）")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}（表达式：{text}）") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("表达式过于复杂，已拒绝计算")

    try:
        result = _eval_node(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("除以零") from exc
    except OverflowError as exc:
        raise ToolError(f"数值溢出：{exc}") from exc
    except ValueError as exc:
        raise ToolError(f"数学域错误：{exc}") from exc

    if isinstance(result, complex):
        raise ToolError("计算结果是复数，计算器只支持实数")
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ToolError(f"计算结果不是有限数：{result}")
    if abs(result) > _MAX_MAGNITUDE:
        raise ToolError("计算结果过大，已拒绝返回")

    return result


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "对数学表达式求值。**任何算术、幂运算、三角函数、对数都应该用它，不要心算。**\n"
        "参数 expression 是一个 Python 风格的数学表达式字符串，例如：\n"
        '  "(1234 * 5678) / 9"、"sqrt(2) * 10"、"2 ** 10"、'
        '"log(100, 10)"、"pi * 3 ** 2"。\n'
        "支持 + - * / // % ** 括号，以及 abs/round/min/max/sum/sqrt/exp/log/log2/log10/"
        "sin/cos/tan/asin/acos/atan/atan2/degrees/radians/floor/ceil/fabs/factorial/gcd/hypot，"
        "常量 pi/e/tau/inf。\n"
        "不支持变量、赋值、自定义函数；一次只算一个表达式，多步计算请分多次调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": '要求值的数学表达式，例如 "(1234 * 5678) / 9"',
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str = "") -> str:
        result = safe_eval(expression)
        # 整数就按整数显示，浮点保留合理精度，避免 0.30000000000000004 这种噪声
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e16:
            return f"{expression.strip()} = {int(result)}"
        if isinstance(result, float):
            return f"{expression.strip()} = {result:.12g}"
        return f"{expression.strip()} = {result}"


def build_tools(config: Any = None) -> list[BaseTool]:
    """工厂函数：签名与其它工具模块保持一致，方便注册中心批量装配。"""
    del config  # 计算器不依赖配置，参数只为统一调用方式
    return [CalculatorTool()]
