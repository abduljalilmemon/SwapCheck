"""Persistent SQLite response caching for LLM provider completions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from swapcheck.providers.base import ChatMessage, LLMResponse


class ResponseCache:
    """Thread-safe SQLite persistent cache for LLM completions."""

    def __init__(self, db_path: Path | str = ".swapcheck_cache.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            if not self.db_path.parent.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS response_cache (
                        cache_key TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER DEFAULT 0
                    );
                    """
                )
                conn.commit()

    @staticmethod
    def compute_key(
        model: str,
        messages: list[ChatMessage] | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Compute deterministic SHA-256 cache key based on invocation inputs."""
        normalized_messages: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                normalized_messages.append(m.to_dict())
            elif isinstance(m, dict):
                normalized_messages.append(m)

        normalized_tools = tools or []
        payload = {
            "model": model,
            "messages": normalized_messages,
            "tools": normalized_tools,
            "temperature": round(temperature, 4),
            "extra": extra or {},
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, cache_key: str) -> LLMResponse | None:
        """Retrieve cached response if present, incrementing hit count."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                "UPDATE response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()

            raw_json = row[0]
            data = json.loads(raw_json)
            resp = LLMResponse.model_validate(data)
            resp.cached = True
            return resp

    def set(self, cache_key: str, model: str, response: LLMResponse) -> None:
        """Store response in cache."""
        # Ensure we don't persist cached flag as True
        resp_dict = response.model_dump()
        resp_dict["cached"] = False
        payload = json.dumps(resp_dict, sort_keys=True)

        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO response_cache (cache_key, model, response_json, hit_count)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json;
                """,
                (cache_key, model, payload),
            )
            conn.commit()

    def stats(self) -> dict[str, int]:
        """Return cache statistics: total entries and total cache hits."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM response_cache;")
            row = cursor.fetchone()
            total_entries = row[0] if row else 0
            total_hits = row[1] if row else 0
            return {"entries": int(total_entries), "hits": int(total_hits)}

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM response_cache;")
            conn.commit()
