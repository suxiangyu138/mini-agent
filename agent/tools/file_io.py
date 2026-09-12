"""文件读写工具。

安全边界：**所有路径都被限制在工作目录（workspace）内**。
模型给的路径会先 resolve 再做前缀校验，``../../`` 这类越界访问会被拒绝。
读操作会限制单次读取长度，避免一个大日志文件直接把上下文冲爆。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 100_000


class _WorkspaceTool(BaseTool):
    """所有文件类工具的公共部分：路径解析 + 沙箱校验。"""

    #: 目录列表时最多返回多少条
    max_entries = 200

    def __init__(self, workspace: str | Path = "./workspace", allow_write: bool = True) -> None:
        self.root = Path(workspace).expanduser().resolve()
        self.allow_write = allow_write
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """把用户路径解析成绝对路径，并确保没跑出工作目录。"""
        raw = (path or "").strip()
        if not raw:
            raise ToolError("路径不能为空")

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ToolError(f"路径 {path!r} 无法解析：{exc}") from exc

        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ToolError(
                f"拒绝访问 {path!r}：超出工作目录 {self.root}。"
                "请使用相对于工作目录的路径，例如 notes/todo.md。"
            )
        return resolved

    def relative(self, path: Path) -> str:
        """统一用正斜杠，这样 Windows 上给模型看的路径和其它平台一致。"""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


class ReadFileTool(_WorkspaceTool):
    name = "read_file"
    #: 文件正文是答案本身，不叠加工具层的长度上限。
    #: 长度由本工具自己的 max_bytes 参数控制，那个上限是显式的、模型能看见也能调大。
    truncate_result = False
    description = (
        "读取工作目录内的文本文件内容。\n"
        "路径相对于工作目录，例如 notes/todo.md 或 data/sales.csv。\n"
        "只能读工作目录内的文件，不能读系统文件。文件过大时会截断并提示。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的文件路径，如 notes/todo.md"},
            "max_bytes": {
                "type": "integer",
                "description": f"最多读取多少字节，默认 {_DEFAULT_MAX_BYTES}",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str = "", max_bytes: int = _DEFAULT_MAX_BYTES) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(self._not_found_hint(target, path))
        if target.is_dir():
            raise ToolError(f"{path!r} 是目录不是文件，请用 list_dir 查看目录内容")

        limit = max(1, int(max_bytes or _DEFAULT_MAX_BYTES))
        size = target.stat().st_size
        try:
            with target.open("rb") as handle:
                raw = handle.read(limit + 1)
        except OSError as exc:
            raise ToolError(f"读取 {path!r} 失败：{exc}") from exc

        truncated = len(raw) > limit or size > limit
        raw = raw[:limit]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # 按字节截断可能刚好把一个多字节字符切成两半，这不是「二进制文件」。
            # 只有确实截断了才容忍残缺的尾巴，否则按二进制拒绝。
            if not truncated:
                raise ToolError(
                    f"{path!r} 不是 UTF-8 文本文件（可能是二进制），共 {size} 字节，无法按文本读取"
                ) from None
            text = raw.decode("utf-8", errors="ignore")

        header = f"文件：{self.relative(target)}（共 {size} 字节）"
        if truncated:
            header += f"\n（内容已截断到前 {limit} 字节，如需后续内容请调大 max_bytes）"
        return f"{header}\n{'-' * 40}\n{text}"

    def _not_found_hint(self, target: Path, raw_path: str) -> str:
        parent = target.parent
        hint = ""
        if parent.exists() and parent.is_dir():
            siblings = sorted(p.name for p in parent.iterdir())[:20]
            if siblings:
                hint = f"该目录下现有：{', '.join(siblings)}"
        if not hint:
            hint = "工作目录下暂无此文件，可先用 list_dir 查看有哪些文件"
        return f"文件不存在：{raw_path!r}。{hint}"


class WriteFileTool(_WorkspaceTool):
    name = "write_file"
    description = (
        "把文本内容写入工作目录内的文件。\n"
        "mode=overwrite 覆盖整个文件（默认），mode=append 追加到文件末尾。\n"
        "父目录不存在时会自动创建。写入前建议先用 read_file 看一眼原内容，避免误覆盖。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的文件路径，如 notes/todo.md"},
            "content": {"type": "string", "description": "要写入的完整文本内容"},
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "overwrite=覆盖（默认），append=追加",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str = "", content: str = "", mode: str = "overwrite") -> str:
        if not self.allow_write:
            raise ToolError("当前配置禁止写文件（allow_file_write=false）")
        if mode not in ("overwrite", "append"):
            raise ToolError(f"mode 只能是 overwrite 或 append，收到 {mode!r}")

        target = self.resolve(path)
        if target.is_dir():
            raise ToolError(f"{path!r} 是已存在的目录，不能写入")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_mode = "a" if mode == "append" else "w"
            with target.open(file_mode, encoding="utf-8", newline="") as handle:
                handle.write(content or "")
        except OSError as exc:
            raise ToolError(f"写入 {path!r} 失败：{exc}") from exc

        action = "追加" if mode == "append" else "写入"
        return (
            f"已{action} {self.relative(target)}：{len(content or '')} 个字符，"
            f"当前文件大小 {target.stat().st_size} 字节"
        )


class ListDirTool(_WorkspaceTool):
    name = "list_dir"
    description = (
        "列出工作目录（或其子目录）下的文件和文件夹，用于确认有哪些文件可用。\n"
        "路径留空则列出工作目录根目录。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的子目录，留空表示根目录"},
        },
        "required": [],
    }

    def run(self, path: str = ".") -> str:
        target = self.resolve(path or ".")
        if not target.exists():
            raise ToolError(f"目录不存在：{path!r}")
        if not target.is_dir():
            raise ToolError(f"{path!r} 是文件不是目录，请用 read_file 读取")

        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"目录 {self.relative(target)} 是空的"

        lines = []
        for entry in entries[: self.max_entries]:
            if entry.is_dir():
                lines.append(f"  [目录] {entry.name}/")
            else:
                lines.append(f"  [文件] {entry.name}  ({entry.stat().st_size} 字节)")
        if len(entries) > self.max_entries:
            lines.append(f"  ……（共 {len(entries)} 项，仅显示前 {self.max_entries} 项）")

        return f"目录 {self.relative(target)} 下有 {len(entries)} 项：\n" + "\n".join(lines)


def build_tools(config: Any = None) -> list[BaseTool]:
    workspace = getattr(config, "workspace_dir", "./workspace")
    allow_write = getattr(config, "allow_file_write", True)
    return [
        ReadFileTool(workspace, allow_write),
        WriteFileTool(workspace, allow_write),
        ListDirTool(workspace, allow_write),
    ]
