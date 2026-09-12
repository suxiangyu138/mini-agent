"""Web 入口层：把同一个 Agent 挂到浏览器上（设计方案 §三.5、§九 第四阶段）。

和最上面那层的关系：**它和 main.py 是平级的两个入口**，谁也不含业务逻辑。
- 装配那一份共用 ``main.build_agent()``，四层里的任何一层都没为网页改过一行；
- 参数解析共用 ``main.cli_overrides()``，「命令行能换模型、网页换不了」这种漂移不会发生。

刻意只用标准库（``http.server`` + SSE）：
本地单机界面为了少写几十行而引一个 Web 框架，换来的是又多一份依赖要维护。

传输用 SSE 而不是 WebSocket 的理由：这一轮对话的数据流本来就是**单向**的
（服务端往外吐字和步骤，客户端只管收），SSE 正好，还自带走神重连以外的全部好处——
浏览器原生支持、纯文本好调试、curl 就能看。

用法::

    python -m web.server                      # http://127.0.0.1:8000
    python -m web.server --port 9000 --open   # 换端口并自动开浏览器
    python -m web.server --provider zhipu     # 换模型后端
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import queue
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 支持 `python web/server.py` 直接跑，不要求先 pip install -e .
    sys.path.insert(0, str(ROOT))

from agent import Agent, AgentResult, StepRecord, __version__  # noqa: E402
from config import Config, setup_logging  # noqa: E402
from main import build_agent, cli_overrides  # noqa: E402
from web import hot  # noqa: E402

logger = logging.getLogger("web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

#: 正文长度上限，防止有人往接口里灌一兆的字符串
MAX_BODY_BYTES = 64 * 1024

#: 本机自己的名字，`Host` 校验用。进程启动时算一次就够。
_LOCAL_NAMES = frozenset(
    n
    for n in (
        "127.0.0.1",
        "::1",
        "localhost",
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
        socket.gethostname().lower().split(".")[0],
    )
    if n
)


def _hostname_of(value: str) -> str:
    """从 ``Host`` / ``Origin`` 里剥出主机名：去端口、去 IPv6 方括号、转小写。

    ``"[::1]:8000"`` → ``"::1"``；``"127.0.0.1:8000"`` → ``"127.0.0.1"``。
    """
    value = (value or "").strip()
    if not value:
        return ""
    return (urlsplit(f"//{value}").hostname or "").lower()


#: 空状态给的建议问题：**每条都挂一个它真正会用到的工具**，工具没注册就不出现。
#: 第一屏就让人看见 ReAct 在干活（真的去调了工具），比任何说明文字都直观。
SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("杭州今天天气怎么样？要带伞吗？", "weather"),
    ("帮我算一下 (1234 * 5678) / 9 是多少", "calculator"),
    ("1 美元等于多少人民币？", "exchange_rate"),
    ("技术圈最近在聊什么？", "hacker_news"),
    ("今年还剩哪些法定假期？", "holidays"),
    # ↓ 以下几条是备选：上面凑不满 _MAX_SUGGESTIONS 条时才会露出来
    ("比特币现在多少钱？", "crypto_price"),
    ("中国最近几年的 GDP 走势如何？", "world_bank"),
    ("今天几号？现在几点？", "get_current_time"),
    ("工作目录里现在有哪些文件？", "list_dir"),
    ("从 2024 年 1 月 1 日到今天过了多少天？", "date_diff"),
    ("最近有什么大模型相关的新闻？", "web_search"),
)

#: 空状态最多摆几条。它们是竖着一列排的，再多就把首屏撑满了——
#: 「高级简洁」的第一条就是别一上来糊一屏字。
_MAX_SUGGESTIONS = 5

#: stop_reason → 给界面看的中文说法
STOP_REASONS = {
    "final": "完成",
    "refusal": "模型拒绝回答",
    "max_steps": "达到步数上限",
    "cancelled": "已中断",
    "error": "出错",
}


# --------------------------------------------------------------------------- #
# 一轮对话：计时与事件出口
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# 会话：一个 Agent + 串行化
# --------------------------------------------------------------------------- #


class ChatSession:
    """网页这一侧的会话。

    Agent 是有状态的（记忆就在它里面），所以同一时刻只能有一轮在跑：
    多开几个标签页也只是排队，不会把两轮对话搅进同一份记忆里。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._agent: Agent | None = None
        self._running: Agent | None = None
        # 下面两个是「当前这一轮」的上下文，由 chat() 装上、跑完摘掉。
        # 核心层的回调是装配时绑死的，所以换轮次只能换它们指向的东西，不能换回调本身。
        self._emit: Callable[[str, dict[str, Any]], None] | None = None
        self._clock: TurnClock | None = None
        self.turns = 0

    # ---------- 装配 ----------

    @property
    def agent(self) -> Agent:
        """懒装配：第一次提问才真的去连模型层，启动因此不受网络影响。"""
        if self._agent is None:
            self._agent = build_agent(self.config, on_text=self._on_text, on_step=self._on_step)
        return self._agent

    def _on_text(self, chunk: str) -> None:
        if self._clock is not None:
            self._clock.mark(chunk)
        if self._emit is not None:
            self._emit("text", {"delta": chunk})

    def _on_step(self, record: StepRecord) -> None:
        # 只推「工具真的被调用」的那一步。final / max_steps 带的是最终答案，它已经走
        # text / done 两条路过去了；再当步骤推一遍，界面上就会多出一个没有名字的工具块。
        if record.kind != "tool" or self._emit is None:
            return
        self._emit("step", step_payload(record))

    # ---------- 控制 ----------

    def cancel(self) -> bool:
        """请求中断正在跑的那一轮。线程安全，随时可调（Agent.cancel 是 Event）。"""
        running = self._running
        if running is None:
            return False
        logger.info("收到中断请求")
        running.cancel()
        return True

    def reset(self) -> None:
        if self._agent is not None:
            self._agent.reset()

    def info(self) -> dict[str, Any]:
        tools = self.agent.tools.names()
        return {
            "version": __version__,
            "provider": self.config.provider,
            "model": self.config.model,
            "tools": tools,
            "max_steps": self.config.max_steps,
            "stream": self.config.stream,
            "workspace": self.config.workspace_dir,
            "turns": self.turns,
            "suggestions": [q for q, tool in SUGGESTIONS if tool in tools][:_MAX_SUGGESTIONS],
        }

    # ---------- 跑一轮 ----------

    def chat(self, message: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """跑一轮对话，把过程中的事件一个个吐出来（SSE 的 event/data 对）。

        生成器是在 HTTP 处理线程里被消费的，Agent 跑在另一个线程：
        这样「一边生成一边推」和「随时能喊停」两件事才可能同时成立——
        要是让处理线程自己去跑 Agent，它在 run() 里出不来，就没人去读中断请求了。
        """
        with self._lock:  # 一轮到底，中途不会被第二个标签页插进来
            agent = self.agent
            events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
            clock = TurnClock(started=time.perf_counter())

            self._emit = lambda name, payload: events.put((name, payload))
            self._clock = clock
            self._running = agent

            box: dict[str, Any] = {}

            def work() -> None:
                try:
                    box["result"] = agent.run(message)
                except Exception as exc:  # 兜底：绝不让异常把一个 HTTP 响应悬在半空
                    logger.exception("这一轮跑挂了")
                    box["error"] = exc
                finally:
                    events.put(None)

            threading.Thread(target=work, daemon=True, name="mini-agent-turn").start()

            try:
                while True:
                    item = events.get()
                    if item is None:
                        break
                    yield item

                if "error" in box:
                    yield "error", {"message": f"{type(box['error']).__name__}: {box['error']}"}
                else:
                    result: AgentResult = box["result"]
                    self.turns += 1
                    yield (
                        "done",
                        {
                            "answer": result.answer,
                            "metrics": metrics_payload(result, clock),
                        },
                    )
            finally:
                self._emit = None
                self._clock = None
                self._running = None
                if "result" not in box and "error" not in box:
                    # 消费者提前跑掉了（关了标签页 / 断了连接）：这一轮没人要了，
                    # 让它在下一个检查点自己收尾，别继续烧 token。
                    logger.info("客户端断开，中止这一轮")
                    agent.cancel()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], session: ChatSession) -> None:
        super().__init__(address, Handler)
        self.session = session


class Handler(BaseHTTPRequestHandler):
    """路由很少，直接用 if 分派，不引路由框架。"""

    server_version = f"MiniAgentWeb/{__version__}"
    protocol_version = "HTTP/1.0"  # SSE 靠「写完后关连接」收尾，正好是 1.0 的语义

    @property
    def session(self) -> ChatSession:
        return self.server.session  # type: ignore[attr-defined]

    # ---------- 同源校验 ----------
    #
    # 这个服务没有登录态，但**「本机 」不等于「只有我能访问」**：
    #   1. DNS rebinding —— 恶意页面把自己的域名解析到 127.0.0.1，浏览器就会带着
    #      攻击者的 Host 来访问本机端口，读走 /api/info、甚至直接驱动 Agent；
    #   2. CSRF —— 页面虽读不到响应，但 ``fetch('http://127.0.0.1:8000/api/chat')``
    #      这种「简单请求」根本不会触发预检，照样能把请求打到我们身上。
    #
    # 两道闸都靠**浏览器自己带的头**，不靠猜：Host 必须落在这台机器真实可被叫到的
    # 名字里；POST 必须同源。两条都不影响 curl（它本来就不是被攻击的目标）。

    def _allowed_hosts(self) -> set[str]:
        """这台机器当前**可以**被叫到的名字。"""
        names = set(_LOCAL_NAMES)
        try:
            # 客户端连过来的那个本机地址。绑 0.0.0.0 时它就是局域网 IP，
            # 于是「局域网用 IP 访问」照常工作，而攻击者的域名仍然对不上。
            names.add(str(self.connection.getsockname()[0]).lower())
        except OSError:
            pass
        return names

    def _guard(self, *, mutating: bool) -> bool:
        """校验通过返回 True；不通过就回 403 并返回 False。"""
        host = _hostname_of(self.headers.get("Host", ""))
        if host not in self._allowed_hosts():
            logger.warning("拒绝 Host=%r 的请求（疑似 DNS rebinding）", host)
            self._send_json({"error": "Host 不被接受"}, status=403)
            return False

        if not mutating:
            return True

        # Sec-Fetch-Site 是新浏览器都会带的，先看它——它比 Origin 更难伪造。
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site and site not in ("same-origin", "none"):
            logger.warning("拒绝 Sec-Fetch-Site=%r 的写请求（疑似 CSRF）", site)
            self._send_json({"error": "跨站请求被拒绝"}, status=403)
            return False

        origin = (self.headers.get("Origin") or "").strip()
        if not origin or origin == "null":
            # 没有 Origin：curl / 脚本，不是浏览器发的跨站请求。
            # 浏览器对跨站 POST **一定**会带 Origin，所以缺它不构成 CSRF 通道。
            return True
        origin_host = _hostname_of(urlsplit(origin).netloc)
        if origin_host and origin_host in self._allowed_hosts():
            return True
        logger.warning("拒绝 Origin=%r 的写请求（疑似 CSRF）", origin)
        self._send_json({"error": "跨站请求被拒绝"}, status=403)
        return False

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # BaseHTTPRequestHandler 规定的名字就是大写下划线
        if not self._guard(mutating=False):
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(INDEX_FILE)
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/") :])
        elif path == "/api/info":
            self._send_json(self.session.info())
        elif path == "/api/hot":
            # 拉不到就返回空 groups，前端会留着静态建议不动。这个接口不报错，
            # 首页的一块装饰不该因为外面某个服务挂了就变成红字。
            self._send_json(hot.groups_with_status())
        elif path == "/favicon.ico":
            self._send_file(STATIC_DIR / "favicon.svg")
        else:
            self._send_json({"error": "没有这个路径"}, status=404)

    def do_POST(self) -> None:
        if not self._guard(mutating=True):
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/chat":
            self._chat()
        elif path == "/api/cancel":
            self._send_json({"cancelled": self.session.cancel()})
        elif path == "/api/reset":
            self.session.reset()
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "没有这个路径"}, status=404)

    # ---------- 各路由的实现 ----------

    def _chat(self) -> None:
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json({"error": "message 不能为空"}, status=400)
            return

        logger.info("提问：%s", message[:120])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")  # 万一前面挂了个会缓冲的反向代理
        self.end_headers()

        try:
            for name, data in self.session.chat(message):
                self._sse(name, data)
        except (BrokenPipeError, ConnectionResetError):
            # 用户关了页面。session.chat 的 finally 会顺手把这轮停掉。
            logger.info("连接已断开")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "文件不存在"}, status=404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")  # 本地开发，改完刷新就见效
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative: str) -> None:
        """静态文件。**必须**校验落点还在 static 目录里，否则就成了任意文件读取。"""
        target = (STATIC_DIR / relative).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self._send_json({"error": "文件不存在"}, status=404)
            return
        self._send_file(target)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _sse(self, name: str, payload: dict[str, Any]) -> None:
        frame = f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()  # 不刷就不是流式了

    # ---------- 日志 ----------

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s %s", self.address_string(), format % args)


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent-web",
        description="Mini-Agent 的网页入口（和 main.py 平级，共用同一份配置与装配）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m web.server                     默认 127.0.0.1:8000\n"
            "  python -m web.server --open              起来就打开浏览器\n"
            "  python -m web.server --provider zhipu    换模型后端\n"
        ),
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，只本机可访问）"
    )
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
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
    parser.add_argument("--log-file", dest="log_file", help="把 DEBUG 日志写到文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="把每一步也打到控制台")
    return parser


def serve(config: Config, host: str, port: int) -> WebServer:
    """起服务（不阻塞）。测试直接拿这个 server 对象跑，不用另起进程。"""
    config.stream = True  # 网页要的就是边生成边看；关掉流式这个界面就没意义了
    session = ChatSession(config)
    server = WebServer((host, port), session)
    logger.info("已监听 http://%s:%d", host, port)
    return server


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = Config.load(args.config_path)
    config.apply_cli_overrides(**cli_overrides(args))
    setup_logging(config.verbose, config.log_file)

    hint = config.missing_key_hint()
    if hint and config.provider != "mock":
        print(f"提示：{hint}", file=sys.stderr)
        print("（可以先跑 python -m web.server --provider mock 离线看看界面）\n", file=sys.stderr)

    try:
        server = serve(config, args.host, args.port)
    except OSError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    # 主页的热点推送要联网、且要几秒，先在后台热上（在打开浏览器之前起跑）。
    # 放在 main 里而不是 serve 里：测试直接调 serve，不该顺带打网络。
    hot.warm()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"Mini-Agent Web · provider={config.provider} · model={config.model}")
    print(f"打开 {url}   （Ctrl+C 停止）")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
