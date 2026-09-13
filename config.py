"""配置管理（设计方案 §六）。

优先级：**环境变量 > 配置文件(config.json) > 代码默认值**。

API Key 绝不出现在代码里：默认从环境变量读，
``config.json`` 也可以写但建议用 ``.env``（已在 .gitignore 里）。

用法::

    cfg = Config.load()                    # 默认路径 ./config.json + ./.env
    cfg = Config.load("my_config.json")    # 指定配置文件
    cfg.apply_cli_overrides(verbose=True)  # 命令行参数优先级最高
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from agent.llm import PROVIDER_PRESETS, resolve_provider

logger = logging.getLogger(__name__)

#: 环境变量前缀，如 MINI_AGENT_MAX_STEPS
ENV_PREFIX = "MINI_AGENT_"

DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_ENV_FILE = ".env"


def load_dotenv(path: str | Path = DEFAULT_ENV_FILE, override: bool = False) -> int:
    """极简 .env 加载器（不想为了这一个功能引入 python-dotenv）。

    默认不覆盖已存在的环境变量 —— 真实环境变量优先级更高。
    返回加载的条目数。
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0

    count = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    if count:
        logger.debug("从 %s 载入 %d 个环境变量", env_path, count)
    return count


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("环境变量 %s=%r 不是整数，用默认值 %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("环境变量 %s=%r 不是数字，用默认值 %s", name, raw, default)
        return default


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("环境变量 %s=%r 不是数字，已忽略", name, raw)
        return None


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return []
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):  # JSON 里的 0 / 1
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on", "y"):
            return True
        if lowered in ("0", "false", "no", "off", "n", ""):
            return False
    raise ValueError(f"要 true 或 false，收到 {value!r}")


def _as_list(value: Any) -> list[str]:
    """列表字段收两种写法：JSON 数组，或者逗号分隔的字符串。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    raise ValueError(f"要一个列表，收到 {value!r}")


def _coerce_config_value(value: Any, current: Any) -> Any:
    """按字段**当前值的类型**收下配置文件里的值。

    这里堵的是一个失败方向：``bool("false")`` 是 ``True``，所以手写配置里
    ``"http_allow_private": "false"``（多打一对引号）会把 SSRF 防线**反向打开**，
    ``"allow_file_write": "false"`` 同理。开关类的错误必须是不生效，不能是反向生效。
    列表也一样：``"http_allowed_hosts": "example.com"`` 会被逐字符当成一个主机名。

    类型推断用当前值而不是注解，是因为 ``from __future__ import annotations``
    下注解是字符串，而且 ``float | None`` 这种联合类型本来也推不出个准数。
    当前值是 ``None`` 的字段（目前只有 temperature）推不出来，原样收下。
    """
    if isinstance(current, bool):
        return _as_bool(value)
    if isinstance(current, list):
        return _as_list(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


@dataclass
class Config:
    """全部配置项。字段名即 config.json 的键名，也是 MINI_AGENT_<大写> 环境变量的来源。"""

    # ---------- 模型层 ----------
    provider: str = "anthropic"
    model: str = ""  # 留空则用 provider 预设的示例模型
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 16000
    # 只有 OpenAI 兼容适配器会发 temperature；Anthropic 开启思考时不接受采样参数
    temperature: float | None = None
    timeout: float = 60.0
    effort: str = ""  # Anthropic: low/medium/high/xhigh/max，留空用服务端默认
    enable_thinking: bool = True  # Anthropic 自适应思考
    show_thinking: bool = False  # 是否把思考摘要打出来
    enable_cache: bool = True  # Anthropic 提示词缓存

    # ---------- 核心层 ----------
    max_steps: int = 8
    max_identical_calls: int = 3  # 同一工具+同一参数超过这个次数就跳过执行
    stream: bool = True

    # ---------- 记忆层 ----------
    max_turns: int = 20
    max_context_chars: int = 60000

    # ---------- 工具层 ----------
    workspace_dir: str = "./workspace"
    allow_file_write: bool = True
    # 工具结果的统一长度上限。这是**兜底**，不是内容工具的主限制：
    # http_request 和 read_file 的正文是答案本身，它们绕开这一项照样整份返回
    # （见 agent/tools/base.py 的 BaseTool.truncate_result），
    # 否则模型会拿着被砍掉一半的正文当完整的用。设 0 表示不限。
    max_tool_result_chars: int = 8000
    http_allow_private: bool = False
    # 白名单：这些域名即使在保留网段也放行（子域名自动跟着放行）。
    # fake-ip 代理环境已由 agent/tools/http.py 自动处理，这里留给「域名合法但确实
    # 解析到内网」的情况。
    http_allowed_hosts: list[str] = field(default_factory=list)
    search_api_key: str = ""
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)

    # ---------- 交互与日志 ----------
    verbose: bool = False
    log_file: str = ""
    system_prompt_extra: str = ""

    # ---------- 网页入口 ----------
    #: 公网访问口令。**建议只放在环境变量 MINI_AGENT_WEB_ACCESS_TOKEN 里**，
    #: 不要写进 config.json——那是个容易被顺手贴出去的文件，而这是个口令。
    #: 留空 = 不做鉴权，只在本机用。一旦设了 web_public_hosts，这一项就是必填：
    #: 「要挂公网」和「没有口令」不能同时成立，serve() 会直接拒绝启动。
    #: 没有命令行参数是故意的：命令行会进 shell 历史和进程列表。
    web_access_token: str = ""
    #: 内网穿透的域名，如 abc123.cpolar.top。写进来的名字才允许通过 Host 校验——
    #: 那道校验默认只认本机名字（防 DNS rebinding），隧道过来的请求 Host 对不上，
    #: 不写在这儿会一路 403。
    web_public_hosts: list[str] = field(default_factory=list)

    # ---------- 内部记账（不对外，不进 config.json） ----------
    #: 上一次套用的 provider 预设名。用来区分 model/base_url 是「预设填的」还是「用户指定的」：
    #: 换 provider 时前者要跟着换，后者必须保留，否则会拿着上个厂商的模型名去请求。
    _preset_applied: str = field(default="", repr=False)
    #: api_key 是从哪个环境变量读来的（如 DEEPSEEK_API_KEY）。换厂商时据此作废旧 Key。
    _key_source: str = field(default="", repr=False)

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls, config_path: str | Path | None = None, env_file: str | Path | None = None
    ) -> Config:
        """按「默认值 → 配置文件 → 环境变量」的顺序构建配置。"""
        config = cls()

        path = Path(config_path) if config_path else Path(DEFAULT_CONFIG_FILE)
        config._apply_file(path)
        load_dotenv(env_file or DEFAULT_ENV_FILE)
        config._apply_env()
        config._resolve_provider_defaults()
        return config

    def _apply_file(self, path: Path) -> None:
        if not path.exists():
            logger.debug("没有配置文件 %s，使用默认值 + 环境变量", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("配置文件 %s 读取失败，已忽略：%s", path, exc)
            return
        if not isinstance(data, dict):
            logger.warning("配置文件 %s 顶层必须是 JSON 对象，已忽略", path)
            return

        known = {item.name for item in fields(self) if not item.name.startswith("_")}
        for key, value in data.items():
            if key.startswith("_"):
                continue  # JSON 没有注释语法，下划线开头的键当注释用
            if key not in known:
                logger.warning("配置文件里有未知配置项 %r，已忽略", key)
                continue
            current = getattr(self, key)
            try:
                setattr(self, key, _coerce_config_value(value, current))
            except (TypeError, ValueError) as exc:
                # 收不下就退回默认值，不把坏值塞进去：宁可让这个开关不生效，
                # 也不能让它带着一个错类型往下跑（bool("false") 那种反向生效）。
                logger.warning("配置项 %r 的值不合适（%s），用默认值 %r", key, exc, current)
        logger.debug("已从 %s 载入配置", path)

    def _apply_env(self) -> None:
        """环境变量优先级最高，覆盖前面所有来源。"""
        env = os.environ
        if env.get(ENV_PREFIX + "PROVIDER"):
            self.provider = env[ENV_PREFIX + "PROVIDER"].strip()
        if env.get(ENV_PREFIX + "MODEL"):
            self.model = env[ENV_PREFIX + "MODEL"].strip()
        if env.get(ENV_PREFIX + "BASE_URL"):
            self.base_url = env[ENV_PREFIX + "BASE_URL"].strip()
        if env.get(ENV_PREFIX + "API_KEY"):
            self.api_key = env[ENV_PREFIX + "API_KEY"].strip()
        if env.get(ENV_PREFIX + "EFFORT"):
            self.effort = env[ENV_PREFIX + "EFFORT"].strip()

        self.max_tokens = _env_int(ENV_PREFIX + "MAX_TOKENS", self.max_tokens)
        self.timeout = _env_float(ENV_PREFIX + "TIMEOUT", self.timeout)
        self.max_steps = _env_int(ENV_PREFIX + "MAX_STEPS", self.max_steps)
        self.max_turns = _env_int(ENV_PREFIX + "MAX_TURNS", self.max_turns)
        self.max_context_chars = _env_int(ENV_PREFIX + "MAX_CONTEXT_CHARS", self.max_context_chars)
        self.max_tool_result_chars = _env_int(
            ENV_PREFIX + "MAX_TOOL_RESULT_CHARS", self.max_tool_result_chars
        )
        self.max_identical_calls = _env_int(
            ENV_PREFIX + "MAX_IDENTICAL_CALLS", self.max_identical_calls
        )
        self.workspace_dir = env.get(ENV_PREFIX + "WORKSPACE", self.workspace_dir)
        self.log_file = env.get(ENV_PREFIX + "LOG_FILE", self.log_file)
        self.system_prompt_extra = env.get(
            ENV_PREFIX + "SYSTEM_PROMPT_EXTRA", self.system_prompt_extra
        )

        temperature = _env_optional_float(ENV_PREFIX + "TEMPERATURE")
        if temperature is not None:
            self.temperature = temperature

        self.verbose = _env_bool(ENV_PREFIX + "VERBOSE", self.verbose)
        self.stream = _env_bool(ENV_PREFIX + "STREAM", self.stream)
        self.enable_thinking = _env_bool(ENV_PREFIX + "ENABLE_THINKING", self.enable_thinking)
        self.show_thinking = _env_bool(ENV_PREFIX + "SHOW_THINKING", self.show_thinking)
        self.enable_cache = _env_bool(ENV_PREFIX + "ENABLE_CACHE", self.enable_cache)
        self.allow_file_write = _env_bool(ENV_PREFIX + "ALLOW_FILE_WRITE", self.allow_file_write)
        self.http_allow_private = _env_bool(
            ENV_PREFIX + "HTTP_ALLOW_PRIVATE", self.http_allow_private
        )

        allowed_hosts = _env_list(ENV_PREFIX + "HTTP_ALLOWED_HOSTS")
        if allowed_hosts:
            self.http_allowed_hosts = allowed_hosts

        token = env.get(ENV_PREFIX + "WEB_ACCESS_TOKEN")
        if token and token.strip():
            self.web_access_token = token.strip()
        public_hosts = _env_list(ENV_PREFIX + "WEB_PUBLIC_HOSTS")
        if public_hosts:
            self.web_public_hosts = public_hosts

        enabled = _env_list(ENV_PREFIX + "ENABLED_TOOLS")
        if enabled:
            self.enabled_tools = enabled
        disabled = _env_list(ENV_PREFIX + "DISABLED_TOOLS")
        if disabled:
            self.disabled_tools = disabled

        # 搜索工具的 Key 用各家惯用的名字，省得用户再包一层
        if not self.search_api_key:
            self.search_api_key = (
                env.get("TAVILY_API_KEY", "") or env.get(ENV_PREFIX + "SEARCH_API_KEY", "")
            ).strip()

    def _resolve_provider_defaults(self) -> None:
        """把 provider 归一化，并补上模型名 / 端点 / API Key 的默认来源。

        这个方法会被调用多次（load 之后、命令行覆盖之后），所以它必须是**幂等**的：
        只填空缺和「上一个预设留下的值」，用户显式写过的配置一律不动。
        """
        try:
            self.provider = resolve_provider(self.provider)
        except Exception as exc:  # 配置错了要给出人话提示，而不是直接崩
            logger.warning("%s，回退到 anthropic", exc)
            self.provider = "anthropic"

        preset = PROVIDER_PRESETS[self.provider]
        previous = PROVIDER_PRESETS.get(self._preset_applied)

        # 用户没写 model / 写的就是上一个预设的默认值 → 换成新预设的（--provider 切换才真的生效）
        if not self.model or (previous and self.model == previous.default_model):
            self.model = preset.default_model
        if not self.base_url or (previous and self.base_url == previous.base_url):
            self.base_url = preset.base_url

        # API Key：换厂商后，从上一家环境变量读来的 Key 就作废了，重新按新厂商的名字找
        if self._key_source and self._key_source != preset.env_key:
            self.api_key = ""
            self._key_source = ""
        if not self.api_key and preset.env_key:
            found = os.environ.get(preset.env_key, "").strip()
            if found:
                self.api_key = found
                self._key_source = preset.env_key

        self._preset_applied = self.provider

    def apply_cli_overrides(self, **overrides: Any) -> Config:
        """命令行参数覆盖（优先级最高）。值为 None 的会被忽略。"""
        known = {item.name for item in fields(self) if not item.name.startswith("_")}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in known:
                logger.debug("忽略未知的命令行配置项 %r", key)
                continue
            setattr(self, key, value)
        self._resolve_provider_defaults()
        return self

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """公开配置项（下划线开头的内部记账字段不对外）。"""
        return {key: value for key, value in asdict(self).items() if not key.startswith("_")}

    def describe(self) -> str:
        """人类可读的配置摘要（--show-config），密钥只显示尾部。"""
        data = self.to_dict()
        # 三个都是秘密，三条都得脱敏。web_access_token 尤其容易漏想：
        # 它跟 API Key 不一样，是**手输**的，多半是人顺手复用的那一串。
        data["api_key"] = _mask(self.api_key)
        data["search_api_key"] = _mask(self.search_api_key)
        data["web_access_token"] = _mask(self.web_access_token)
        width = max(len(key) for key in data)
        return "\n".join(f"  {key.ljust(width)} : {value!r}" for key, value in data.items())

    def missing_key_hint(self) -> str:
        """启动时如果缺 Key，返回一句人话提示；不缺则返回空串。"""
        preset = PROVIDER_PRESETS.get(self.provider)
        if preset is None or not preset.env_key:
            return ""
        if self.api_key:
            return ""
        return (
            f"当前 provider={self.provider} 需要 API Key，但没有读到。\n"
            f"请设置环境变量 {preset.env_key}"
            f"（或 {ENV_PREFIX}API_KEY，或写在 .env / config.json 里）。"
        )


def _mask(secret: str) -> str:
    if not secret:
        return "(未设置)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}（共 {len(secret)} 位）"


def setup_logging(verbose: bool = False, log_file: str = "") -> None:
    """配置日志：控制台按 verbose 决定级别，文件永远记 DEBUG（可追溯，§三.4）。"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    # 第三方库的 DEBUG 太吵，压到 WARNING
    for noisy in ("urllib3", "httpx", "httpcore", "anthropic", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
