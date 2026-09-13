"""入口层：交互循环，最薄的一层（设计方案 §三.5）。

这里**不写任何业务逻辑**，只做四件事：
读用户输入 → 交给 Agent → 打印结果 → 处理退出/中断。
正因如此，将来换成 Web / API 只要替换这个文件，下面四层一行都不用动。

用法::

    python main.py                       # 交互模式
    python main.py -q "1234 * 5678 等于多少"   # 单次提问
    python main.py --demo                # 离线演示（不需要 API Key）
    python main.py --list-tools          # 看有哪些工具
    python main.py --list-models         # 问厂商要可用模型清单（顺便验 Key）
    python main.py --show-config         # 看当前生效的配置
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from agent import Agent, AgentResult, Memory, StepRecord, build_default_registry, create_llm
from agent.prompt import build_system_prompt
from config import Config, setup_logging

logger = logging.getLogger("main")

BANNER = r"""
  __  __ _       _         _                    _
 |  \/  (_)_ __ (_)       / \   __ _  ___ _ __ | |_
 | |\/| | | '_ \| |_____ / _ \ / _` |/ _ \ '_ \| __|
 | |  | | | | | | |_____/ ___ \ (_| |  __/ | | | |_
 |_|  |_|_|_| |_|_|    /_/   \_\__, |\___|_| |_|\__|
                               |___/
"""

EXIT_WORDS = {"exit", "quit", "q", "bye", "退出", "再见", "结束"}


# --------------------------------------------------------------------------- #
# 展示：把 StepRecord 画成人看得懂的样子（可观测性，§三.4）
# --------------------------------------------------------------------------- #


def print_step(record: StepRecord) -> None:
    """打印一步的中间过程（verbose 模式下由 Agent 回调）。"""
    if record.kind == "tool":
        icon = "✗" if record.is_error else "→"
        args = _compact_args(record.arguments)
        head = f"  {icon} [{record.index}] {record.tool_name}({args})"
        if record.elapsed:
            head += f"  {record.elapsed:.2f}s"
        print(head, file=sys.stderr)
        body = record.result if record.is_error else _clip(record.result, 300)
        for line in str(body).splitlines() or [""]:
            print(f"      {line}", file=sys.stderr)
    elif record.kind == "error":
        print(f"  ! [{record.index}] {record.text}", file=sys.stderr)
    elif record.kind == "max_steps":
        print("  ! 达到最大步数，强制结束", file=sys.stderr)


def _compact_args(arguments: dict[str, Any]) -> str:
    if not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        text = str(value)
        text = text if len(text) <= 60 else text[:60] + "…"
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


def _clip(text: str, limit: int) -> str:
    flat = str(text)
    return flat if len(flat) <= limit else flat[:limit] + "…"


def list_models(config: Config) -> int:
    """问厂商要一份可用模型清单。模型换代快，猜不如问，顺便也验了 Key 通不通。"""
    try:
        llm = create_llm(config)
        models = llm.list_models()
    except Exception as exc:
        print(f"取模型列表失败：{exc}", file=sys.stderr)
        return 2
    if not models:
        print(f"{config.provider} 这家没有提供模型列表接口，直接填 model 名即可。")
        return 0
    print(f"{config.provider} 当前可用 {len(models)} 个模型：")
    for name in models:
        mark = "  ← 当前使用" if name == config.model else ""
        print(f"  {name}{mark}")
    return 0


def print_result(agent: Agent, result: AgentResult, verbose: bool) -> None:
    """打印结果。流式模式下面向用户的文字已经边收边打过了，这里只补个换行。"""
    printer = agent.on_text
    if isinstance(printer, StreamPrinter) and printer.wrote:
        printer.close_line()
        printer.reset()
    else:
        print(result.answer)
    if verbose:
        usage = result.usage
        meta = (
            f"  ── {result.steps} 步 · {result.stop_reason} · "
            f"输入 {usage.get('input_tokens', 0)} / 输出 {usage.get('output_tokens', 0)} tokens"
        )
        if usage.get("cache_read_input_tokens"):
            meta += f" · 命中缓存 {usage['cache_read_input_tokens']}"
        print(meta, file=sys.stderr)


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #


class StreamPrinter:
    """流式回调：模型一边生成一边打。

    两个状态位各管一件事：``wrote`` 说明这一轮的字已经打出去了（最后别再重复打印答案），
    ``_open`` 说明光标停在一个还没换行的行尾（日志插进来之前得先收掉这一行）。
    """

    def __init__(self) -> None:
        self.wrote = False
        self._open = False

    def __call__(self, chunk: str) -> None:
        self.wrote = True
        self._open = True
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def close_line(self) -> None:
        """结束当前流式输出这一行（没开过头就什么都不做）。"""
        if self._open:
            print()
            sys.stdout.flush()  # 输出被重定向时 stdout 是块缓冲的，这里主动刷一下，免得日志插到前面
            self._open = False

    def reset(self) -> None:
        self.wrote = False
        self._open = False


class Reporter:
    """把「步骤过程」和「流式输出」凑到一起，让两者不会挤在同一行上。

    流式输出是边收边打的，末尾没有换行；此时 stderr 上的日志（比如 core 的「循环结束」）
    插进来就会接在答案屁股后面。所以拿到最终答案的那一刻先把这一行收掉。
    """

    def __init__(self, verbose: bool, printer: StreamPrinter | None = None) -> None:
        self.verbose = verbose
        self.printer = printer

    def __call__(self, record: StepRecord) -> None:
        if record.kind in ("final", "max_steps") and self.printer is not None:
            self.printer.close_line()
        if self.verbose:
            print_step(record)


def build_agent(
    config: Config,
    on_text: Callable[[str], None] | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
) -> Agent:
    """按配置装配四层。**换模型只改这一处调用**（create_llm 内部查表）。

    ``on_text`` / ``on_step`` 不传就是 CLI 的默认行为（边收边打 + verbose 打印）；
    Web 入口把自己的回调传进来，装配逻辑一行都不用抄第二遍（§三.5「换 UI 只换入口层」）。
    """
    llm = create_llm(config)
    tools = build_default_registry(config)
    system_prompt = build_system_prompt(
        tools=tools.names(),
        workspace=config.workspace_dir if "read_file" in tools else "",
        extra=config.system_prompt_extra,
    )
    memory = Memory(
        system_prompt=system_prompt,
        max_turns=config.max_turns,
        max_context_chars=config.max_context_chars,
    )

    if on_text is None and on_step is None:
        printer = StreamPrinter() if config.stream else None
        on_text = printer
        on_step = Reporter(config.verbose, printer)

    return Agent(
        llm=llm,
        tools=tools,
        memory=memory,
        config=config,
        on_step=on_step,
        on_text=on_text,
    )


#: 命令行里能直接覆盖的配置项。CLI 和 Web 两个入口共用这一份，避免两边漂移
#: （否则容易出现「命令行能换模型、网页换不了」这种说不清的差异）。
CLI_CONFIG_KEYS = (
    "provider",
    "model",
    "base_url",
    "api_key",
    "max_steps",
    "max_turns",
    "workspace_dir",
    "log_file",
)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """把解析好的命令行参数翻成 ``config.apply_cli_overrides`` 认识的字典。"""
    overrides: dict[str, Any] = {key: getattr(args, key, None) for key in CLI_CONFIG_KEYS}
    stream = getattr(args, "stream", None)
    if stream is not None:
        overrides["stream"] = stream
    if getattr(args, "verbose", False):
        overrides["verbose"] = True
    # --public-host 是 append 出来的列表，落到 web_public_hosts 上。
    # 口令刻意没有对应的命令行参数：那会进 shell 历史和进程列表。
    public_hosts = getattr(args, "public_host", None)
    if public_hosts:
        overrides["web_public_hosts"] = public_hosts
    return overrides


COMMANDS = {
    "/help": "显示帮助",
    "/tools": "列出当前可用工具",
    "/reset": "清空对话历史（保留系统提示）",
    "/config": "显示当前配置",
    "/exit": "退出",
}


def print_help() -> None:
    print("命令：")
    for name, desc in COMMANDS.items():
        print(f"  {name:<10} {desc}")
    print("直接输入内容即可提问，Ctrl+C 中断当前输入，再按一次退出。")


def handle_command(line: str, agent: Agent, config: Config) -> bool:
    """处理斜杠命令。返回 True 表示要继续循环。"""
    command = line.strip().lower()
    if command in ("/exit", "/quit"):
        return False
    if command == "/help":
        print_help()
    elif command == "/tools":
        print(f"当前 {len(agent.tools)} 个工具：")
        print(agent.tools.describe())
    elif command == "/reset":
        agent.reset()
        print("对话历史已清空。")
    elif command == "/config":
        print("当前配置：")
        print(config.describe())
    else:
        print(f"未知命令 {line}，输入 /help 查看可用命令。")
    return True


# --------------------------------------------------------------------------- #
# 交互
# --------------------------------------------------------------------------- #


def run_once(agent: Agent, question: str, verbose: bool) -> int:
    result = agent.run(question)
    print_result(agent, result, verbose)
    return 0 if result.ok else 1


def run_interactive(agent: Agent, config: Config) -> int:
    print(f"Mini-Agent 已就绪 · provider={config.provider} · model={config.model}")
    tools = agent.tools.names()
    print(f"已加载 {len(tools)} 个工具：{', '.join(tools) or '（无）'}")
    print("输入 /help 看命令，输入 exit 退出。\n")

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            print("再见。")
            return 0
        if line.startswith("/"):
            if not handle_command(line, agent, config):
                print("再见。")
                return 0
            continue

        try:
            result = agent.run(line)
        except KeyboardInterrupt:
            print("\n（已中断本轮对话，历史保留）")
            continue
        print_result(agent, result, config.verbose)
        print()


DEMO_QUESTION = "帮我算一下 (1234 * 5678) / 9 是多少，顺便看下今天几号。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="一个可运行、可扩展的通用 Agent（ReAct + 工具调用 + 可插拔模型层）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py                            交互模式\n"
            '  python main.py -q "今天几号"               单次提问\n'
            "  python main.py --demo                     离线演示，不需要 API Key\n"
            "  python main.py --provider deepseek        换模型后端\n"
        ),
    )
    parser.add_argument("-q", "--query", help="单次提问后退出")
    parser.add_argument(
        "-c", "--config", dest="config_path", help="配置文件路径（默认 config.json）"
    )
    parser.add_argument(
        "--provider",
        help="模型后端：anthropic/openai/deepseek/qwen/moonshot/zhipu/minimax/mimo/ollama/mock…",
    )
    parser.add_argument("--model", help="模型名，覆盖配置")
    parser.add_argument("--base-url", dest="base_url", help="自定义 API 端点")
    parser.add_argument("--api-key", dest="api_key", help="API Key（建议用环境变量，不要写在这里）")
    parser.add_argument("--max-steps", dest="max_steps", type=int, help="ReAct 最大步数（默认 8）")
    parser.add_argument(
        "--max-turns", dest="max_turns", type=int, help="记忆保留的最大轮数（默认 20）"
    )
    parser.add_argument("--workspace", dest="workspace_dir", help="文件工具的工作目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每一步的中间过程")
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false", default=None, help="关闭流式输出"
    )
    parser.add_argument("--log-file", dest="log_file", help="把 DEBUG 日志写到文件")
    parser.add_argument("--list-tools", action="store_true", help="列出工具后退出")
    parser.add_argument(
        "--list-models", action="store_true", help="问厂商要一份可用模型清单后退出（顺便验 Key）"
    )
    parser.add_argument("--show-config", action="store_true", help="显示生效配置后退出")
    parser.add_argument("--demo", action="store_true", help="用 mock 模型离线演示多步工具调用")
    return parser


def _force_utf8_console() -> None:
    """Windows 控制台默认可能是 GBK，中文输出会炸，这里尽量切成 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()

    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config.load(args.config_path)
    overrides = cli_overrides(args)
    if args.demo:
        # 离线演示：强制走 mock，顺便把详细过程打开
        overrides.update(provider="mock", verbose=True, stream=True)
    config.apply_cli_overrides(**overrides)

    setup_logging(config.verbose, config.log_file)

    if args.show_config:
        print("当前生效配置：")
        print(config.describe())
        return 0

    if args.list_models:
        return list_models(config)

    try:
        agent = build_agent(config)
    except Exception as exc:  # 装配失败（多半是缺依赖或缺 Key）要给人话提示
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    if args.list_tools:
        print(f"当前 {len(agent.tools)} 个工具：")
        print(agent.tools.describe())
        return 0

    if args.query:
        return run_once(agent, args.query, config.verbose)

    if args.demo:
        print(BANNER)
        print(f"离线演示 · 问题：{DEMO_QUESTION}\n")
        return run_once(agent, DEMO_QUESTION, verbose=True)

    hint = config.missing_key_hint()
    if hint and config.provider != "mock":
        print(f"提示：{hint}", file=sys.stderr)
        print("（可以先跑 python main.py --demo 离线看看效果）\n", file=sys.stderr)

    print(BANNER)
    return run_interactive(agent, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        raise SystemExit(130) from None
