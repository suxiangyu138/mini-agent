"""记忆层：管理对话历史，控制上下文长度。

对应设计方案 §三.3。核心是**裁剪时不能把 assistant 的工具调用和对应的 tool 结果拆散**
——它们必须成对保留或成对删除，否则消息格式非法，下一轮请求直接报错。

这里的做法是把历史切成「调用组」，再把调用组归入以 user 消息为界的「轮」，
裁剪永远以**整轮**为单位丢弃，从根上避免拆散配对。

进阶方向（§三.3）留了接口但没实现：传入 ``compressor`` 即可在裁剪时把
被丢掉的历史交给它压缩成摘要（见 :class:`Compressor`）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Protocol

from .llm import ToolCall, assistant_message, system_message, tool_message, user_message

logger = logging.getLogger(__name__)

#: 角色常量
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"

Message = dict[str, Any]
#: 调用组：assistant 的工具调用 + 紧随其后的 tool 结果
Group = list[Message]
#: 轮：一条 user 消息开启一轮，包含该轮里所有调用组
Turn = list[Group]


class Compressor(Protocol):
    """摘要压缩接口（§三.3 进阶方向，先留接口不实现）。

    裁剪发生时，被丢弃的整轮历史会交给 ``compress()``；
    返回若干条消息（通常是「用户问了什么 + 助手总结」）插到保留历史之前。
    返回空列表表示这次不压缩，那些消息就真的丢了。
    """

    def compress(self, messages: list[Message]) -> list[Message]: ...


def split_into_groups(messages: Iterable[Message]) -> list[Group]:
    """把消息切成「调用组」：assistant 的工具调用 + 紧随其后的 tool 结果算一组。"""
    groups: list[Group] = []
    for message in messages:
        previous = groups[-1] if groups else None
        if (
            message.get("role") == TOOL
            and previous
            and previous[0].get("role") == ASSISTANT
            and previous[0].get("tool_calls")
        ):
            previous.append(message)
        else:
            groups.append([message])
    return groups


def split_into_turns(messages: list[Message]) -> list[Turn]:
    """把调用组归入「轮」：一条 user 消息开启一轮，直到下一条 user 消息为止。"""
    turns: list[Turn] = []
    for group in split_into_groups(messages):
        if group[0].get("role") == USER or not turns:
            turns.append([group])
        else:
            turns[-1].append(group)
    return turns


class Memory:
    """对话记忆。

    对外只暴露「追加 / 读取 / 清空」三个动作，裁剪在读取时自动发生。
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_turns: int = 20,
        max_context_chars: int = 60000,
        compressor: Compressor | None = None,
    ) -> None:
        self.system_prompt = system_prompt or ""
        self.max_turns = max(1, int(max_turns))
        #: 历史字符数上限（粗粒度估算，不引第三方 tokenizer；字符数 ≈ token 数的 1~2 倍）
        self.max_context_chars = max(0, int(max_context_chars))
        self.compressor = compressor
        self._messages: list[dict[str, Any]] = []
        self.trim_count = 0

    # ---------- 追加 ----------

    def add_user(self, text: str) -> dict[str, Any]:
        return self._append(user_message(text))

    def add_assistant(
        self,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        raw: Any = None,
        provider: str = "",
    ) -> dict[str, Any]:
        return self._append(assistant_message(content, tool_calls, raw=raw, provider=provider))

    def add_tool_result(
        self, tool_call_id: str, name: str, content: str, is_error: bool = False
    ) -> dict[str, Any]:
        return self._append(tool_message(tool_call_id, name, content, is_error))

    def _append(self, message: dict[str, Any]) -> dict[str, Any]:
        self._messages.append(message)
        return message

    # ---------- 读取 ----------

    def get_messages(self) -> list[dict[str, Any]]:
        """返回给模型的消息列表：system + （裁剪后的）历史。

        裁剪只在这里发生，所以调用方拿到的永远是一份合法的历史。
        """
        history = self._trim()
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(system_message(self.system_prompt))
        messages.extend(history)
        return messages

    def raw_messages(self) -> list[dict[str, Any]]:
        """未经裁剪的完整历史（调试 / 落盘用）。"""
        return list(self._messages)

    # ---------- 清空 ----------

    def clear(self) -> None:
        """保留 system，清掉其余（§三.3）。"""
        self._messages.clear()
        logger.debug("记忆已清空（system 提示保留）")

    # ---------- 裁剪 ----------

    def _trim(self) -> list[Message]:
        if not self._messages:
            return []

        turns = split_into_turns(self._messages)
        dropped: list[Turn] = []

        # 1) 按轮数裁剪
        while len(turns) > self.max_turns:
            dropped.append(turns.pop(0))

        # 2) 按字符预算裁剪（从最旧的整轮开始丢，永远保留最后一轮 = 当前问题）
        if self.max_context_chars:
            while len(turns) > 1 and self._count_chars(turns) > self.max_context_chars:
                dropped.append(turns.pop(0))

        if dropped:
            self.trim_count += len(dropped)
            logger.info("上下文超限，已丢弃 %d 轮历史（当前保留 %d 轮）", len(dropped), len(turns))

        history: list[Message] = [message for turn in turns for group in turn for message in group]

        # 3) 兜底：消息列表必须以 user 开头（各家 API 的硬要求）
        while history and history[0].get("role") != USER:
            history.pop(0)

        # 4) 有 compressor 就把丢掉的历史压缩成摘要插回去
        if dropped and self.compressor is not None:
            history = [*self._compressed_prefix(dropped), *history]

        return history

    def _compressed_prefix(self, dropped: list[Turn]) -> list[Message]:
        flat: list[Message] = [message for turn in dropped for group in turn for message in group]
        try:
            summary = self.compressor.compress(flat)  # type: ignore[union-attr]
        except Exception:  # 压缩失败不该影响主流程，退化成「真的丢掉」
            logger.exception("历史压缩失败，本次退化为直接丢弃")
            return []

        if not summary:
            return []
        # 摘要必须以 user 开头，否则格式非法
        if summary[0].get("role") != USER:
            summary = [user_message("（以下是更早对话的摘要）"), *summary]
        return summary

    @staticmethod
    def _count_chars(turns: list[Turn]) -> int:
        total = 0
        for turn in turns:
            for group in turn:
                for message in group:
                    total += len(str(message.get("content") or ""))
                    for call in message.get("tool_calls") or []:
                        total += len(str(call.arguments))
        return total

    # ---------- 观测 ----------

    def stats(self) -> dict[str, Any]:
        turns = split_into_turns(self._messages)
        return {
            "messages": len(self._messages),
            "turns": len(turns),
            "chars": self._count_chars(turns),
            "trimmed_turns": self.trim_count,
        }

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:  # pragma: no cover
        info = self.stats()
        return f"<Memory messages={info['messages']} turns={info['turns']} chars={info['chars']}>"
