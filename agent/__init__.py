"""Mini-Agent 核心包。

四层结构（设计方案 §二），自下而上：

- :mod:`agent.llm`    模型层：统一接口 + 各厂商适配器
- :mod:`agent.tools`  工具层：工具基类 + 注册中心 + 内置工具
- :mod:`agent.memory` 记忆层：对话历史与裁剪
- :mod:`agent.core`   核心层：ReAct 主循环（唯一同时用到上面三层的模块）

入口层（main.py）只 import 这个包里导出的名字，不直接碰具体适配器。
"""

from __future__ import annotations

from .core import Agent, AgentResult, StepRecord
from .llm import (
    PROVIDER_PRESETS,
    BaseLLM,
    LLMError,
    LLMResponse,
    ToolCall,
    create_llm,
    resolve_provider,
)
from .memory import Compressor, Memory
from .prompt import SYSTEM_PROMPT, build_system_prompt
from .tools import BaseTool, ToolError, ToolRegistry, ToolResult, build_default_registry

__version__ = "0.1.0"

__all__ = [
    # 核心层
    "Agent",
    "AgentResult",
    "StepRecord",
    # 模型层
    "BaseLLM",
    "LLMError",
    "LLMResponse",
    "ToolCall",
    "PROVIDER_PRESETS",
    "create_llm",
    "resolve_provider",
    # 记忆层
    "Memory",
    "Compressor",
    # 工具层
    "BaseTool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    # 提示词
    "SYSTEM_PROMPT",
    "build_system_prompt",
    "__version__",
]
