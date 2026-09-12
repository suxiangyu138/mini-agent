"""纯娱乐工具（Dog CEO，免 Key）。

它在这套工具层里的位置有点特别：**唯一一个不产出信息的工具**。
存在的理由是个人 Agent 不该只会干活——用户说「给我看张狗的照片」时，
它得真的去取一张，而不是回一句「我没有这个能力」。

返回的是图片 URL。网页界面会把 markdown 图片渲染出来，
命令行界面则只显示链接——两种情况都要能用，所以链接一定给全。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolError
from .net import get_json, tool_timeout

logger = logging.getLogger(__name__)

_BASE = "https://dog.ceo/api"


class DogImageTool(BaseTool):
    name = "dog_image"
    description = (
        "随机取一张狗的图片，返回可直接打开的 URL。\n"
        "**用户说「给我看张狗的照片」「来张狗狗图」这类请求时用它**，"
        "不要说自己不能发图片——把返回的链接用 markdown 图片语法贴出来就行。\n"
        "可以指定品种（英文，如 husky、corgi、pug），不指定就随机。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "breed": {
                "type": "string",
                "description": "狗的品种，英文小写（husky、corgi、pug、labrador…）；省略则随机",
            },
        },
        "required": [],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def run(self, breed: str = "") -> str:
        name = (breed or "").strip().lower().replace(" ", "")
        if not name:
            data = get_json(f"{_BASE}/breeds/image/random", timeout=self.timeout, service="Dog CEO")
            return self._format("随机", data)

        if "-" in name:  # 子品种要拆成路径：terrier-scottish → terrier/scottish
            parent, _, child = name.partition("-")
            path = f"{parent}/{child}"
        else:
            path = name

        try:
            data = get_json(
                f"{_BASE}/breed/{path}/images/random",
                timeout=self.timeout,
                service="Dog CEO",
            )
        except ToolError as exc:
            raise ToolError(
                f"没有叫「{name}」的品种（{exc}）。用英文小写品种名，"
                "比如 husky、corgi、pug、beagle、labrador、retriever-golden；"
                "不指定品种就直接随机。"
            ) from exc
        return self._format(name, data)

    @staticmethod
    def _format(breed: str, data: dict[str, Any]) -> str:
        url = data.get("message")
        if not url or data.get("status") != "success":
            raise ToolError("Dog CEO 没有返回图片地址，稍后再试")
        return (
            f"品种：{breed}\n"
            f"图片：{url}\n"
            f"markdown：![dog]({url})\n"
            "（把这行 markdown 原样贴出来，网页界面就会显示这张图）"
        )


def build_tools(config: Any = None) -> list[BaseTool]:
    """免 Key，默认就有。"""
    return [DogImageTool(timeout=tool_timeout(config))]
