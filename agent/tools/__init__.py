"""工具包：内置工具 + 默认装配。

新增一个工具只要两步（设计方案 §十一 验收标准 3）：
  1. 在本目录新建一个文件，写一个继承 :class:`BaseTool` 的类，填好三要素；
  2. 在下面的 ``_BUILDERS`` 里加上该模块的 ``build_tools``。

工具模块统一暴露 ``build_tools(config) -> list[BaseTool]``，
需要配置的（文件目录、API Key）从 config 里取，不需要的忽略即可。
"""

from __future__ import annotations

import logging
from typing import Any

from . import (
    academic,
    books,
    calculator,
    crypto,
    datetime_tool,
    exchange_rate,
    file_io,
    fun,
    holidays,
    http,
    search,
    tech_news,
    weather,
    world_bank,
)
from .base import BaseTool, ToolError, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]

#: 内置工具模块。想默认启用新工具，把它加进来就行。
#:
#: 后九个是「公共 API」那一组：全部**免 Key**，装好就能用，
#: 主机名在各自模块里写死，模型没法把请求指到别处（见 :mod:`agent.tools.net`）。
#: 每个工具的 Schema 都会跟着**每一次**请求发给模型，所以这一组的粒度是刻意压过的——
#: 「OpenAlex / Crossref / PubMed」合成一个 ``academic_search`` 换 ``source`` 参数，
#: 就是为了少带两份 Schema。再加工具前先想想能不能并进现有的。
_BUILDERS = (
    calculator,
    datetime_tool,
    file_io,
    http,
    search,
    # ---- 公共 API（免 Key）----
    weather,
    academic,
    world_bank,
    crypto,
    exchange_rate,
    tech_news,
    books,
    holidays,
    fun,
)


def build_default_registry(config: Any = None) -> ToolRegistry:
    """按配置装配默认工具集。

    ``config.disabled_tools`` 里的名字会被剔除，
    ``config.enabled_tools`` 非空时则只保留其中的名字。
    """
    tools: list[BaseTool] = []
    for module in _BUILDERS:
        try:
            tools.extend(module.build_tools(config))
        except Exception:  # 单个工具模块出问题不该拖垮整个启动
            logger.exception("装配工具模块 %s 失败，已跳过", module.__name__)

    disabled = set(getattr(config, "disabled_tools", None) or [])
    enabled = set(getattr(config, "enabled_tools", None) or [])
    if enabled:
        tools = [tool for tool in tools if tool.name in enabled]
    if disabled:
        tools = [tool for tool in tools if tool.name not in disabled]

    registry = ToolRegistry(
        tools,
        max_result_chars=int(getattr(config, "max_tool_result_chars", 8000)),
    )
    logger.debug("已装配 %d 个工具：%s", len(registry), ", ".join(registry.names()))
    return registry
