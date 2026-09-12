"""记忆的落盘层：会话、消息、长期事实、滚动摘要、设置。

**为什么是 SQLite。** 设计方案里那一层写的是「短期记忆用 Redis、长期记忆用 SQLite/PostgreSQL」。
Redis 是个要单独跑起来的服务，而这个项目从第一天起就只有一条依赖线——``pyproject.toml``
的 dependencies 里只有 ``anthropic`` 和 ``requests``。为了一个本机单机 Agent 要求用户
先起一个 Redis，代价和收益完全不成比例。``sqlite3`` 是标准库，单文件、零运维，
恰好接住「窗口在内存里、长期记忆落盘」这个分工：进程活着时历史就在 :class:`~agent.memory.Memory`
里，这一层只负责**跨会话、跨重启**那部分。

**线程模型。** ``ThreadingHTTPServer`` 每个请求开一个线程，而 sqlite3 的连接默认不许跨线程用。
这里用 ``check_same_thread=False`` 加一把可重入锁把**一个**连接共享出去，而不是「每次操作
开一个新连接」——后者在线程一多就把打开/关闭的开销乘上去，WAL 的好处也一并丢掉。
锁是粗粒度的，但这儿根本没有并发压力：本机界面的写入频率是「人打字的速度」。

**只存必要的东西。** 落盘的消息里会剥掉 ``_provider_raw``（厂商原始 content 块，
里面可能带着几千 token 的 thinking）。它是「同厂商下一轮复用」的优化，不是历史本身；
存进去只会让库文件涨得飞快，还顺带把模型的思考过程永久留在磁盘上。重启后适配器
会从 text + tool_calls 重建，功能不受影响，只是少了一点复用。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .llm import ToolCall, assistant_message, tool_message, user_message

logger = logging.getLogger(__name__)

#: 事实的分类。顺序即界面上的展示顺序（标签页从左到右）。
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("identity", "身份"),
    ("preference", "偏好"),
    ("project", "项目"),
    ("other", "其他"),
)
CATEGORY_KEYS = tuple(key for key, _ in CATEGORIES)
CATEGORY_LABELS = dict(CATEGORIES)


def category_label(key: str) -> str:
    return CATEGORY_LABELS.get(key, CATEGORY_LABELS["other"])


# --------------------------------------------------------------------------- #
# 长期记忆的一条事实
# --------------------------------------------------------------------------- #


@dataclass
class Fact:
    """一条长期记忆。

    ``version`` 和 ``updated_at`` 不是装饰：用户的偏好会变（「预算 5000」变成「预算 8000」），
    没有时间维度就会出现新旧两条同时生效、模型看到自相矛盾的上下文。
    改动时版本号 +1、时间戳刷新，界面上能显示「更新于」，注入时也能只取最新的那一条。
    """

    id: str
    content: str
    category: str = "other"
    source_session: str = ""
    source_title: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    disabled: bool = False
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category_label"] = category_label(self.category)
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Fact:
        return cls(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            source_session=row["source_session"],
            source_title=row["source_title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            disabled=bool(row["disabled"]),
            version=row["version"],
        )


# --------------------------------------------------------------------------- #
# 消息的序列化
# --------------------------------------------------------------------------- #


def dump_message(message: dict[str, Any]) -> str:
    """消息 → JSON 文本。剥掉 ``_provider_raw``，理由见模块文档。"""
    calls = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, ToolCall):
            calls.append(asdict(call))
        elif isinstance(call, dict):  # 已经是 dict 的就别动它
            calls.append(call)
    payload = {
        "role": message.get("role", ""),
        "content": message.get("content", ""),
        "tool_calls": calls,
    }
    for key in ("tool_call_id", "name", "is_error"):
        if key in message:
            payload[key] = message[key]
    return json.dumps(payload, ensure_ascii=False)


def load_message(raw: str) -> dict[str, Any] | None:
    """JSON 文本 → 消息。坏数据返回 ``None``，由调用方跳过。

    一条读不出来的记录不该让整个会话打不开——历史是「有更好、没有也能继续」的东西。
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("会话里有一条消息读不出来，已跳过")
        return None
    if not isinstance(data, dict):
        return None

    role = str(data.get("role") or "")
    calls = [
        ToolCall(
            id=str(call.get("id") or ""),
            name=str(call.get("name") or ""),
            arguments=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
            raw_arguments=str(call.get("raw_arguments") or ""),
            parse_error=str(call.get("parse_error") or ""),
        )
        for call in data.get("tool_calls") or []
        if isinstance(call, dict)
    ]

    if role == "user":
        return user_message(str(data.get("content") or ""))
    if role == "assistant":
        return assistant_message(str(data.get("content") or ""), calls)
    if role == "tool":
        return tool_message(
            str(data.get("tool_call_id") or ""),
            str(data.get("name") or ""),
            str(data.get("content") or ""),
            bool(data.get("is_error")),
        )
    return None


# --------------------------------------------------------------------------- #
# 库
# --------------------------------------------------------------------------- #

#: 当前 schema 版本，写在 ``PRAGMA user_version`` 里。
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    turns       INTEGER NOT NULL DEFAULT 0,
    inherit_memory INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    session_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY,
    content     TEXT NOT NULL DEFAULT '',
    upto        INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
    id             TEXT PRIMARY KEY,
    content        TEXT NOT NULL,
    category       TEXT NOT NULL DEFAULT 'other',
    source_session TEXT NOT NULL DEFAULT '',
    source_title   TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    disabled       INTEGER NOT NULL DEFAULT 0,
    version        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    """默认库文件位置。跟 ``workspace/`` 一样是运行产物，已在 .gitignore 里。"""
    return Path("./data/memory.db")


class Store:
    """一堆会话 + 一堆事实，落在同一个 SQLite 文件里。

    所有方法都是线程安全的（内部一把可重入锁）。时间戳一律由调用方传入或在这里取
    ``time.time()``——测试需要能把时钟摆到「3 天前」，所以凡是涉及时间的公开方法
    都留了 ``now`` 参数。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path if path is not None else default_db_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # 外键默认是关的（SQLite 的历史包袱），不打开的话 ON DELETE CASCADE 形同虚设，
            # 删掉会话之后消息会一直留在库里。
            self._conn.execute("PRAGMA foreign_keys = ON")
            # WAL：读不挡写。这一层没多少并发，但成本是零，且能让「正在生成时刷新页面」
            # 这种读操作不必等写事务。
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #

    def create_session(
        self,
        session_id: str,
        title: str = "",
        inherit_memory: bool = True,
        now: float | None = None,
    ) -> dict[str, Any]:
        stamp = time.time() if now is None else now
        self._write(
            "INSERT INTO sessions"
            " (id, title, created_at, updated_at, pinned, turns, inherit_memory)"
            " VALUES (?, ?, ?, ?, 0, 0, ?)",
            (session_id, title, stamp, stamp, int(bool(inherit_memory))),
        )
        record = self.get_session(session_id)
        assert record is not None  # 刚插进去的
        return record

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(rows[0]) if rows else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """置顶的排前面，其余按最近活动时间倒序。界面按这个顺序铺。"""
        rows = self._rows("SELECT * FROM sessions ORDER BY pinned DESC, updated_at DESC")
        return [dict(row) for row in rows]

    def touch_session(
        self,
        session_id: str,
        turns: int | None = None,
        title: str | None = None,
        now: float | None = None,
    ) -> None:
        """一次对话之后刷新活动时间（顺带把轮数记上，供列表里显示）。"""
        stamp = time.time() if now is None else now
        with self._lock:
            if title is not None:
                self._conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
                )
            if turns is not None:
                self._conn.execute(
                    "UPDATE sessions SET turns = ? WHERE id = ?", (turns, session_id)
                )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (stamp, session_id)
            )
            self._conn.commit()

    def rename_session(self, session_id: str, title: str) -> None:
        self._write("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    def set_pinned(self, session_id: str, pinned: bool) -> None:
        self._write("UPDATE sessions SET pinned = ? WHERE id = ?", (int(bool(pinned)), session_id))

    def delete_session(self, session_id: str) -> None:
        self._write("DELETE FROM sessions WHERE id = ?", (session_id,))

    def delete_all_sessions(self) -> int:
        """清空所有会话，返回删掉的条数。

        只删会话——消息和摘要靠外键级联跟着走，而**长期记忆是另一张表，不在这里被牵连**。
        这一条是刻意的：清空对话记录和抹掉「你是谁」是两件事，后者要单独确认。
        """
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            self._conn.execute("DELETE FROM sessions")
            self._conn.commit()
        return int(count)

    # ------------------------------------------------------------------ #
    # 消息
    # ------------------------------------------------------------------ #

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """整体重写某个会话的消息。

        不搞增量追加：历史一旦被裁剪或压缩，顺序就不再是「只往后加」了，
        增量写迟早会和内存里的那份对不上。会话最多几百条，整体重写的代价可以忽略，
        换来的是「库里的历史 == 内存里的历史」这个不需要论证的不变式。
        """
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.executemany(
                "INSERT INTO messages (session_id, seq, data) VALUES (?, ?, ?)",
                [(session_id, index, dump_message(m)) for index, m in enumerate(messages)],
            )
            self._conn.commit()

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT data FROM messages WHERE session_id = ? ORDER BY seq", (session_id,)
        )
        messages = []
        for row in rows:
            message = load_message(row["data"])
            if message is not None:
                messages.append(message)
        return messages

    def message_count(self, session_id: str) -> int:
        rows = self._rows("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,))
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ #
    # 滚动摘要
    # ------------------------------------------------------------------ #

    def get_summary(self, session_id: str) -> tuple[str, int]:
        """返回 ``(摘要正文, 覆盖到第几条消息)``。

        第二个值是水位线：它之前的历史已经被压进摘要里了，不必重复压。
        没有摘要时返回 ``("", 0)``。
        """
        rows = self._rows("SELECT content, upto FROM summaries WHERE session_id = ?", (session_id,))
        if not rows:
            return "", 0
        return str(rows[0]["content"]), int(rows[0]["upto"])

    def set_summary(self, session_id: str, content: str, upto: int) -> None:
        self._write(
            "INSERT INTO summaries (session_id, content, upto, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET content = excluded.content,"
            " upto = excluded.upto, updated_at = excluded.updated_at",
            (session_id, content, int(upto), time.time()),
        )

    def clear_summary(self, session_id: str) -> None:
        self._write("DELETE FROM summaries WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------ #
    # 长期记忆
    # ------------------------------------------------------------------ #

    def list_facts(self, include_disabled: bool = True) -> list[Fact]:
        """按「最近更新在前」排。注入给模型时只取启用中的，界面则要看得见被禁用的。

        末尾那个 ``id`` 是**决胜键**，不是排序依据：``time.time()`` 在 Windows 上
        只有约 15ms 的粒度，而抽取器一次能返回好几条——同一批写进来的事实时间戳
        常常一模一样。只按 ``updated_at`` 排的话，这几条之间的先后由 SQLite 自由发挥，
        同一个列表每渲染一次就可能换个顺序。加上 ``id`` 只为让结果**稳定**。
        """
        sql = "SELECT * FROM facts"
        if not include_disabled:
            sql += " WHERE disabled = 0"
        sql += " ORDER BY updated_at DESC, id DESC"
        return [Fact.from_row(row) for row in self._rows(sql)]

    def get_fact(self, fact_id: str) -> Fact | None:
        rows = self._rows("SELECT * FROM facts WHERE id = ?", (fact_id,))
        return Fact.from_row(rows[0]) if rows else None

    def add_fact(
        self,
        fact_id: str,
        content: str,
        category: str = "other",
        source_session: str = "",
        source_title: str = "",
        now: float | None = None,
    ) -> Fact:
        stamp = time.time() if now is None else now
        self._write(
            "INSERT INTO facts (id, content, category, source_session, source_title,"
            " created_at, updated_at, disabled, version) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1)",
            (fact_id, content, category, source_session, source_title, stamp, stamp),
        )
        fact = self.get_fact(fact_id)
        assert fact is not None
        return fact

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        disabled: bool | None = None,
        now: float | None = None,
    ) -> Fact | None:
        """改一条事实。内容真的变了才抬版本号——只是禁用/启用不算「改过」。"""
        current = self.get_fact(fact_id)
        if current is None:
            return None
        stamp = time.time() if now is None else now
        changed = content is not None and content != current.content
        if category is not None and category != current.category:
            changed = True

        with self._lock:
            self._conn.execute(
                "UPDATE facts SET content = ?, category = ?, disabled = ?, updated_at = ?,"
                " version = version + ? WHERE id = ?",
                (
                    current.content if content is None else content,
                    current.category if category is None else category,
                    int(current.disabled if disabled is None else bool(disabled)),
                    stamp if changed or disabled is not None else current.updated_at,
                    int(changed),
                    fact_id,
                ),
            )
            self._conn.commit()
        return self.get_fact(fact_id)

    def delete_fact(self, fact_id: str) -> None:
        self._write("DELETE FROM facts WHERE id = ?", (fact_id,))

    def delete_facts_from_session(self, session_id: str) -> int:
        """删掉「这个会话产生的那几条记忆」。界面上的删除确认里有这么一个开关。"""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE source_session = ?", (session_id,)
            ).fetchone()[0]
            self._conn.execute("DELETE FROM facts WHERE source_session = ?", (session_id,))
            self._conn.commit()
        return int(count)

    def delete_all_facts(self) -> int:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            self._conn.execute("DELETE FROM facts")
            self._conn.commit()
        return int(count)

    # ------------------------------------------------------------------ #
    # 设置
    # ------------------------------------------------------------------ #

    def get_settings(self) -> dict[str, Any]:
        rows = self._rows("SELECT key, value FROM settings")
        settings: dict[str, Any] = {}
        for row in rows:
            try:
                settings[row["key"]] = json.loads(row["value"])
            except (TypeError, ValueError):
                logger.warning("设置项 %r 读不出来，已忽略", row["key"])
        return settings

    def set_settings(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in values.items()],
            )
            self._conn.commit()


# --------------------------------------------------------------------------- #
# 时间：界面上那些「3 天前」
# --------------------------------------------------------------------------- #


def humanize_age(stamp: float, now: float | None = None) -> str:
    """把时间戳说成人话（「刚刚」「3 天前」）。

    放在服务端而不是前端：前端算的话，浏览器时区和服务端不一致时会出现
    「3 分钟前」和「今天」两个分组对不上号的怪事。**分组和时间用同一个时钟。**
    """
    if not stamp:
        return ""
    seconds = max(0.0, (time.time() if now is None else now) - stamp)
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    if seconds < 86400 * 30:
        return f"{int(seconds // 86400)} 天前"
    if seconds < 86400 * 365:
        return f"{int(seconds // (86400 * 30))} 个月前"
    return f"{int(seconds // (86400 * 365))} 年前"
