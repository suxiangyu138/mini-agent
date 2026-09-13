"""Web 入口层：把 Agent 挂到浏览器上（设计方案 §三.5、§九 第四阶段）。

和最上面那层的关系：**它和 main.py 是平级的两个入口**，谁也不含业务逻辑。
- 装配那一份共用 ``main.build_agent()``，四层里的任何一层都没为网页改过一行；
- 参数解析共用 ``main.cli_overrides()``，「命令行能换模型、网页换不了」这种漂移不会发生。

**这一层只做三件事**：路由、同源校验、把会话层吐出来的事件写成 SSE。
「一个会话是什么」「记忆怎么抽」这类问题都在 :mod:`web.sessions` 里回答，
这里连一次 ``store`` 都不直接调。

刻意只用标准库（``http.server`` + SSE）：
本地单机界面为了少写几十行而引一个 Web 框架，换来的是又多一份依赖要维护。

传输用 SSE 而不是 WebSocket 的理由：这一轮对话的数据流本来就是**单向**的
（服务端往外吐字和步骤，客户端只管收），SSE 正好，还自带走神重连以外的全部好处——
浏览器原生支持、纯文本好调试、curl 就能看。

用法::

    python -m web.server                      # http://127.0.0.1:8000
    python -m web.server --port 9000 --open   # 换端口并自动开浏览器
    python -m web.server --provider zhipu     # 换模型后端

**挂公网**（内网穿透，比如 cpolar）要同时给两样东西：

    set MINI_AGENT_WEB_ACCESS_TOKEN=<一串随机字符>
    python -m web.server --public-host abc123.cpolar.top

前者是口令，后者告诉服务「这个域名是合法的入口」——不写的话隧道过来的请求会
因为 Host 对不上被 403（那道校验本来是防 DNS rebinding 的）。**两样必须一起给**：
只给域名不给口令，serve() 会直接拒绝启动。理由是这个服务没有登录态时，拿到
URL 的人可以花你的 API 额度、翻你的对话和长期记忆、让 Agent 读写工作目录。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import mimetypes
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 支持 `python web/server.py` 直接跑，不要求先 pip install -e .
    sys.path.insert(0, str(ROOT))

from agent import __version__  # noqa: E402
from agent.store import Store  # noqa: E402
from config import Config, setup_logging  # noqa: E402
from main import cli_overrides  # noqa: E402
from web import hot  # noqa: E402
from web.sessions import Session, SessionError, SessionManager  # noqa: E402
from web.telemetry import (  # noqa: E402,F401  ← 这几个名字从这里重新导出给调用方用
    STOP_REASONS,
    TurnClock,
    metrics_payload,
    step_payload,
)

logger = logging.getLogger("web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
LOGIN_FILE = STATIC_DIR / "login.html"

#: 正文长度上限，防止有人往接口里灌一兆的字符串
MAX_BODY_BYTES = 64 * 1024

# --------------------------------------------------------------------------- #
# 口令闸
# --------------------------------------------------------------------------- #

#: cookie 名。带上 mini_agent 前缀是因为挂公网时这个域名下可能不止我们一个页面，
#: 用 token / pass 这种名字撞车的概率不低。
_COOKIE_NAME = "mini_agent_pass"

#: cookie 里放的是**口令的 HMAC，不是口令本身**。截图、共享屏幕、贴日志都会漏出
#: cookie；而人往往到处用同一个口令，漏出去就不是这一个服务的事了。
#: 标签写死在这里：它不参与保密，只是让这个值换个用途就对不上。
_COOKIE_LABEL = b"mini-agent-web-v1"
_COOKIE_MAX_AGE = 30 * 24 * 3600

#: 口令试错的容忍度。**计数是全体的，不分来源**——穿透之后所有请求都是从本机
#: 转过来的，按 IP 分等于没分。反正这服务就一个用户，锁就锁全体。
#: 试满之后歇一分钟：挡不住决心，但足够让「一串随机字符」这件事本身成为门槛。
_MAX_FAILURES = 10
_LOCKOUT_SECONDS = 60
_FAILURE_DELAY = 0.4

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


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        manager: SessionManager,
        *,
        access_token: str = "",
        public_hosts: list[str] | None = None,
    ) -> None:
        super().__init__(address, Handler)
        self.manager = manager
        self.access_token = (access_token or "").strip()
        self.public_hosts = [h.strip().lower().rstrip(".") for h in (public_hosts or []) if h]
        #: 口令试错计数：{来源: [次数, 最后一次的时间]}。线程池里多个请求会同时碰它。
        self.login_failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def note_login_failure(self, who: str) -> None:
        with self._lock:
            record = self.login_failures.setdefault(who, [0.0, 0.0])
            record[0] += 1
            record[1] = time.time()

    def login_locked(self, who: str) -> bool:
        with self._lock:
            record = self.login_failures.get(who)
            if not record or record[0] < _MAX_FAILURES:
                return False
            if time.time() - record[1] > _LOCKOUT_SECONDS:
                del self.login_failures[who]  # 锁过期，重新给机会
                return False
            return True

    def note_login_success(self, who: str) -> None:
        with self._lock:
            self.login_failures.pop(who, None)

    @property
    def session(self) -> Session:
        """当前会话。写成属性而不是启动时抓一份：

        「当前会话」是会变的（新建 / 切换 / 删掉当前那个），启动时存下来的那份
        在用户点一下侧栏之后就成了幽灵——接口还往旧会话里写。
        """
        return self.manager.current()


class Handler(BaseHTTPRequestHandler):
    """路由很少，直接用 if 分派，不引路由框架。"""

    server_version = f"MiniAgentWeb/{__version__}"
    protocol_version = "HTTP/1.0"  # SSE 靠「写完后关连接」收尾，正好是 1.0 的语义

    @property
    def manager(self) -> SessionManager:
        return self.server.manager  # type: ignore[attr-defined]

    @property
    def session(self) -> Session:
        return self.manager.current()

    @property
    def token(self) -> str:
        """配置里的口令。空串 = 不做鉴权（只在本机用的那种用法）。"""
        return self.server.access_token  # type: ignore[attr-defined]

    @property
    def public_hosts(self) -> list[str]:
        return self.server.public_hosts  # type: ignore[attr-defined]

    # ---------- 三道闸 ----------
    #
    # **「本机」不等于「只有我能访问」**，挂了穿透更不等于：
    #   1. Host —— 恶意页面把自己的域名解析到 127.0.0.1（DNS rebinding），浏览器
    #      就会带着攻击者的 Host 来访问本机端口，读走 /api/info、甚至驱动 Agent；
    #   2. Origin —— CSRF。页面虽读不到响应，但 ``fetch('/api/chat')`` 这种
    #      「简单请求」根本不触发预检，照样能把请求打到我们身上；
    #   3. 口令 —— 前两道挡的是**别的网页借你的浏览器**，挡不住「知道 URL 的人
    #      自己发请求」。挂公网必须补这一道。
    #
    # 前两道靠**浏览器自己带的头**，不靠猜：Host 必须落在真实可被叫到的名字里，
    # POST 必须同源。两条都不影响 curl（它本来就不是被攻击的目标）。
    # 第三道配了口令才生效，本机用法一行都不用改。
    #
    # 新增的写接口全部走 ``mutating=True`` 这一条——这就意味着**任何**改状态的请求
    # 都被这几道闸罩住，不存在「忘了加校验」的新接口。

    def _allowed_hosts(self) -> set[str]:
        """这台机器当前**可以**被叫到的名字。"""
        names = set(_LOCAL_NAMES)
        names.update(self.public_hosts)  # 穿透域名：是「我同意挂出去的入口」，不是猜的
        try:
            # 客户端连过来的那个本机地址。绑 0.0.0.0 时它就是局域网 IP，
            # 于是「局域网用 IP 访问」照常工作，而攻击者的域名仍然对不上。
            names.add(str(self.connection.getsockname()[0]).lower())
        except OSError:
            pass
        return names

    def _guard(self, *, mutating: bool, path: str = "") -> bool:
        """校验通过返回 True；不通过就自己回掉响应并返回 False。"""
        host = _hostname_of(self.headers.get("Host", ""))
        if host not in self._allowed_hosts():
            logger.warning("拒绝 Host=%r 的请求（疑似 DNS rebinding）", host)
            self._send_json({"error": "Host 不被接受"}, status=403)
            return False

        if not self._authorized(path):
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

    # ---------- 口令 ----------

    def _cookie_value(self) -> str:
        """cookie 该带的值：口令的 HMAC（理由见 ``_COOKIE_LABEL`` 上面那段）。"""
        return hmac.new(self.token.encode("utf-8"), _COOKIE_LABEL, hashlib.sha256).hexdigest()

    def _cookies(self) -> dict[str, str]:
        jar: dict[str, str] = {}
        for chunk in (self.headers.get("Cookie") or "").split(";"):
            key, sep, value = chunk.partition("=")
            if sep:
                jar[key.strip()] = value.strip()
        return jar

    def _authorized(self, path: str) -> bool:
        """没配口令一律放行；配了就要求 cookie 对得上。"""
        if not self.token or path in ("/login", "/api/login"):
            return True  # 登录页本身要是也被拦，就没人进得来了
        given = self._cookies().get(_COOKIE_NAME, "")
        if given and hmac.compare_digest(given, self._cookie_value()):
            return True
        if path.startswith("/api/"):
            # 前端据此跳登录页；报 403 会让它显示成「未连接」，把话说不清楚。
            self._send_json({"error": "需要先登录", "login": "/login"}, status=401)
        else:
            self._redirect("/login")
        return False

    def _cookie_header(self) -> str:
        parts = [
            f"{_COOKIE_NAME}={self._cookie_value()}",
            "Path=/",
            f"Max-Age={_COOKIE_MAX_AGE}",
            "HttpOnly",  # 页面脚本读不到，XSS 也偷不走
            "SameSite=Lax",  # 别的站点发起的请求带不上它，CSRF 少一条路
        ]
        if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https":
            # 只在确实是 https 时加：本地 http 访问加了它，浏览器会直接不存这个 cookie。
            parts.append("Secure")
        return "; ".join(parts)

    def _login(self) -> None:
        """表单提交口令。故意用表单而不是 JS：这样整条链路不依赖前端脚本。"""
        if not self.token:
            self._redirect("/")
            return
        who = self.address_string()
        if self.server.login_locked(who):  # type: ignore[attr-defined]
            logger.warning("口令试错次数超限，暂时拒绝（来自 %s）", who)
            self._redirect("/login?error=locked")
            return

        provided = self._read_form().get("token", "")
        if provided and hmac.compare_digest(provided, self.token):
            self.server.note_login_success(who)  # type: ignore[attr-defined]
            logger.info("口令验证通过（来自 %s）", who)
            self._redirect("/", cookie=self._cookie_header())
            return

        self.server.note_login_failure(who)  # type: ignore[attr-defined]
        logger.warning("口令不对（来自 %s）", who)
        time.sleep(_FAILURE_DELAY)
        self._redirect("/login?error=1")

    def _logout(self) -> None:
        # Max-Age=0 就是让浏览器立刻丢掉它。值给空串，不给 HMAC——
        # 万一有浏览器只认值不认 Max-Age，留下的也是个对不上的空值。
        self._send_json(
            {"ok": True},
            extra=(("Set-Cookie", f"{_COOKIE_NAME}=; Path=/; Max-Age=0"),),
        )

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return {}
        try:
            raw = self.rfile.read(length).decode("utf-8")
        except UnicodeDecodeError:
            return {}
        return {key: values[0] for key, values in parse_qs(raw).items() if values}

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # BaseHTTPRequestHandler 规定的名字就是大写下划线
        path = self.path.split("?", 1)[0]
        if not self._guard(mutating=False, path=path):
            return
        if path == "/login":
            if self.token:
                self._send_file(LOGIN_FILE)
            else:
                # 没配口令就没有登录这回事，别给一个永远登不进去的页面。
                self._redirect("/")
        elif path in ("/", "/index.html"):
            self._send_file(INDEX_FILE)
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/") :])
        elif path == "/favicon.ico":
            self._send_file(STATIC_DIR / "favicon.svg")
        elif path == "/api/info":
            self._send_json(self._info())
        elif path == "/api/hot":
            # 拉不到就返回空 groups，前端会留着静态建议不动。这个接口不报错，
            # 首页的一块装饰不该因为外面某个服务挂了就变成红字。
            self._send_json(hot.groups_with_status())
        elif path == "/api/sessions":
            self._send_json(self.manager.list_sessions())
        elif path == "/api/session":
            self._send_json(self.session.detail())
        elif path == "/api/context":
            self._send_json(self.session.context())
        elif path == "/api/memories":
            self._send_json(self.manager.list_facts())
        elif path == "/api/settings":
            self._send_json(self.manager.settings_payload())
        else:
            self._send_json({"error": "没有这个路径"}, status=404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._guard(mutating=True, path=path):
            return
        if path == "/api/chat":
            self._chat()
            return
        if path == "/api/login":
            self._login()  # 表单，不是 JSON：这里自己读 body，别落到下面去
            return
        if path == "/api/logout":
            self._logout()
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        # 下面这些都会改状态，全部包在一处 try 里：会话层抛 SessionError 就是
        # 用户填错了（4xx），别的异常让它照常炸成 500——那才是程序写错了。
        try:
            if path == "/api/cancel":
                self._send_json({"cancelled": self.session.cancel()})
            elif path == "/api/reset":
                self.session.reset()
                self._send_json({"ok": True, "session": self.session.detail()})
            elif path == "/api/sessions/new":
                inherit = payload.get("inherit")
                session = self.manager.new(None if inherit is None else bool(inherit))
                self._send_json(session.detail())
            elif path == "/api/sessions/select":
                session = self.manager.select(str(payload.get("id") or ""))
                self._send_json(session.detail())
            elif path == "/api/sessions/rename":
                self.manager.rename(str(payload.get("id") or ""), str(payload.get("title") or ""))
                self._send_json(self.manager.list_sessions())
            elif path == "/api/sessions/pin":
                self.manager.pin(str(payload.get("id") or ""), bool(payload.get("pinned")))
                self._send_json(self.manager.list_sessions())
            elif path == "/api/sessions/delete":
                result = self.manager.delete(
                    str(payload.get("id") or ""), bool(payload.get("with_memories"))
                )
                self._send_json({**result, "sessions": self.manager.list_sessions()})
            elif path == "/api/sessions/clear":
                count = self.manager.clear_all(str(payload.get("confirm") or ""))
                self._send_json({"cleared": count, "sessions": self.manager.list_sessions()})
            elif path == "/api/memories/add":
                fact = self.manager.add_fact(
                    str(payload.get("content") or ""),
                    str(payload.get("category") or "other"),
                )
                self._send_json({"fact": fact.to_dict(), "memories": self.manager.list_facts()})
            elif path == "/api/memories/update":
                fact = self.manager.update_fact(
                    str(payload.get("id") or ""),
                    content=payload.get("content"),
                    category=payload.get("category"),
                    disabled=payload.get("disabled"),
                )
                self._send_json({"fact": fact.to_dict(), "memories": self.manager.list_facts()})
            elif path == "/api/memories/delete":
                self.manager.delete_fact(str(payload.get("id") or ""))
                self._send_json({"memories": self.manager.list_facts()})
            elif path == "/api/memories/clear":
                self._send_json(
                    {"cleared": self.manager.clear_facts(), "memories": self.manager.list_facts()}
                )
            elif path == "/api/settings":
                # 收两种写法：``{"values": {...}}`` 和直接 ``{key: value}``。
                # 前端那一个地方顺手写成后者是很自然的事，为此回一个 400 属于自找麻烦。
                values = payload.get("values")
                self._send_json(
                    self.manager.update_settings(values if isinstance(values, dict) else payload)
                )
            else:
                self._send_json({"error": "没有这个路径"}, status=404)
        except SessionError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception:
            logger.exception("处理 %s 时出错", path)
            self._send_json({"error": "服务端错误"}, status=500)

    # ---------- 各路由的实现 ----------

    def _info(self) -> dict[str, Any]:
        session = self.session
        tools = session.agent.tools.names()
        return {
            "version": __version__,
            "provider": self.manager.config.provider,
            "model": self.manager.config.model,
            "tools": tools,
            "max_steps": self.manager.config.max_steps,
            "stream": self.manager.config.stream,
            "workspace": self.manager.config.workspace_dir,
            "suggestions": [q for q, tool in SUGGESTIONS if tool in tools][:_MAX_SUGGESTIONS],
            "session": session.brief(),
            "settings": self.manager.settings,
            # 没配口令就别显示「退出登录」——那是个点了也没意义的按钮
            "auth": bool(self.token),
        }

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

        # 允许指定会话：页面上开着的是哪个会话就发到哪个。不传则用当前的。
        # 少了这个，两个标签页开着不同会话时，后发的那个会悄悄写进另一个会话里。
        wanted = str(payload.get("session") or "")
        try:
            session = self.manager.select(wanted) if wanted else self.session
        except SessionError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        logger.info("提问（会话 %s）：%s", session.id[:8], message[:120])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")  # 万一前面挂了个会缓冲的反向代理
        self.end_headers()

        try:
            for name, data in session.chat(message):
                self._sse(name, data)
        except (BrokenPipeError, ConnectionResetError):
            # 用户关了页面。session.chat 的 finally 会顺手把这轮停掉。
            logger.info("连接已断开")

    def _send_json(
        self,
        payload: dict[str, Any],
        status: int = 200,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, cookie: str | None = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

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
            "\n"
            "挂公网（内网穿透）：\n"
            "  set MINI_AGENT_WEB_ACCESS_TOKEN=<一串随机字符>\n"
            "  python -m web.server --public-host abc123.cpolar.top\n"
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
    parser.add_argument("--db", dest="db_path", help="会话与记忆的库文件（默认 data/memory.db）")
    parser.add_argument(
        "--public-host",
        dest="public_host",
        action="append",
        metavar="域名",
        help=(
            "内网穿透的公网域名，可重复。写进来的名字才允许通过 Host 校验；"
            "给了它就必须同时设 MINI_AGENT_WEB_ACCESS_TOKEN，否则拒绝启动。"
            "口令本身故意没有命令行参数——那会进 shell 历史和进程列表"
        ),
    )
    parser.add_argument("--log-file", dest="log_file", help="把 DEBUG 日志写到文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="把每一步也打到控制台")
    return parser


def serve(config: Config, host: str, port: int, store: Store | None = None) -> WebServer:
    """起服务（不阻塞）。测试直接拿这个 server 对象跑，不用另起进程。

    ``store`` 是留给测试的口子：单元测试要的是一份干净的、彼此隔离的库，
    不能让他们往开发者本机那个 ``data/memory.db`` 里写东西。生产路径传 None，
    由 :class:`~agent.store.Store` 自己决定落在哪。

    口令和公网域名从 ``config`` 读（命令行覆盖和环境变量在更早的地方已经汇进
    config 了），这里只负责那条不变式：**要挂公网，就必须有口令**。
    """
    config.stream = True  # 网页要的就是边生成边看；关掉流式这个界面就没意义了

    token = str(getattr(config, "web_access_token", "") or "").strip()
    public_hosts = [str(h).strip() for h in (getattr(config, "web_public_hosts", None) or []) if h]

    # 「要挂公网」和「没有口令」不能同时成立。宁可起不来，也别把一个没有登录态、
    # 还能花你 API 额度的界面挂出去——那件事一旦发生是收不回来的。
    if public_hosts and not token:
        raise ValueError(
            "配了公网域名（web_public_hosts / --public-host）却没有口令："
            "请先设置环境变量 MINI_AGENT_WEB_ACCESS_TOKEN 再启动"
        )
    if token and len(token) < 12:
        logger.warning("访问口令只有 %d 个字符，偏短；建议至少 16 位随机字符", len(token))

    manager = SessionManager(config, store)
    server = WebServer((host, port), manager, access_token=token, public_hosts=public_hosts)
    if public_hosts:
        logger.info("已放行公网域名：%s（口令闸已开）", ", ".join(public_hosts))
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
        server = serve(config, args.host, args.port, Store(args.db_path))
    except (OSError, ValueError) as exc:
        # ValueError 是「要挂公网却没给口令」这类配置错误，同样是一句话说得清的。
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    # 主页的热点推送要联网、且要几秒，先在后台热上（在打开浏览器之前起跑）。
    # 放在 main 里而不是 serve 里：测试直接调 serve，不该顺带打网络。
    hot.warm()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"Mini-Agent Web · provider={config.provider} · model={config.model}")
    print(f"打开 {url}   （Ctrl+C 停止）")
    for public in server.public_hosts:
        # 口令一个字都不打。终端会被录屏、会被截图，而它就在上面几行之外。
        print(f"公网入口 https://{public}/   先过口令闸（MINI_AGENT_WEB_ACCESS_TOKEN）")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
        server.manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
