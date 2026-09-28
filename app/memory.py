"""Memory for the companion, isolated per customer (user_id).

Layers
------
conversation : LLM message history (Bedrock Converse format) + rolling summary +
               a clean UI transcript. Compacted when long.
session      : structured working memory for the current conversation - goal,
               budget, preferences, rejections, presented option lists (for
               "the first option"), pending checkout, last order.
long_term    : facts that survive "New conversation" - stated preferences,
               dislikes, past goals, orders.
events       : audit log of tool actions (shown in the UI / useful for the demo video).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

EMPTY_SESSION = {
    "goal": None, "budget": None, "budget_scope": "per_item", "preferences": [], "dislikes": [],
    "rejected": {}, "presented": [], "pending_checkout": None, "last_order": None, "notes": [], "ui_events": [],
}
EMPTY_LONG_TERM = {"preferences": [], "dislikes": [], "goals": [], "orders": []}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path):
        self.path = str(path)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations(user_id TEXT PRIMARY KEY, messages TEXT NOT NULL,
                transcript TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', turn INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT);
            CREATE TABLE IF NOT EXISTS session_memory(user_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS long_term_memory(user_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT,
                kind TEXT, payload TEXT);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def lock(self, user_id) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(user_id, threading.Lock())

    # ---------------- conversation
    def load_conversation(self, user_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM conversations WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return {"messages": [], "transcript": [], "summary": "", "turn": 0}
        return {"messages": json.loads(row["messages"]), "transcript": json.loads(row["transcript"]),
                "summary": row["summary"], "turn": row["turn"]}

    def save_conversation(self, user_id, conv):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?,?,?,?,?,?)",
                       (user_id, json.dumps(conv["messages"]), json.dumps(conv["transcript"][-200:]),
                        conv.get("summary", ""), conv.get("turn", 0), _now()))

    # ---------------- structured memory
    def load_session(self, user_id):
        with self._db() as db:
            row = db.execute("SELECT data FROM session_memory WHERE user_id=?", (user_id,)).fetchone()
        data = json.loads(row["data"]) if row else {}
        return {**json.loads(json.dumps(EMPTY_SESSION)), **data}

    def save_session(self, user_id, data):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO session_memory VALUES (?,?)", (user_id, json.dumps(data)))

    def load_long_term(self, user_id):
        with self._db() as db:
            row = db.execute("SELECT data FROM long_term_memory WHERE user_id=?", (user_id,)).fetchone()
        data = json.loads(row["data"]) if row else {}
        return {**json.loads(json.dumps(EMPTY_LONG_TERM)), **data}

    def save_long_term(self, user_id, data):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO long_term_memory VALUES (?,?)", (user_id, json.dumps(data)))

    # ---------------- events
    def log(self, user_id, kind, payload):
        with self._db() as db:
            db.execute("INSERT INTO events(user_id, ts, kind, payload) VALUES (?,?,?,?)",
                       (user_id, _now(), kind, json.dumps(payload, default=str)[:4000]))

    def recent_events(self, user_id, limit=50):
        with self._db() as db:
            rows = db.execute("SELECT ts, kind, payload FROM events WHERE user_id=? ORDER BY id DESC LIMIT ?",
                              (user_id, limit)).fetchall()
        return [{"ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows]

    # ---------------- reset
    def reset_conversation(self, user_id):
        """New conversation: clears history + working memory; keeps long-term memory, basket and orders."""
        with self._db() as db:
            db.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            db.execute("DELETE FROM session_memory WHERE user_id=?", (user_id,))
        self.log(user_id, "reset", {"at": time.time()})

    def forget_everything(self, user_id):
        self.reset_conversation(user_id)
        with self._db() as db:
            db.execute("DELETE FROM long_term_memory WHERE user_id=?", (user_id,))
