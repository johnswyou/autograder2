"""The call cache: a response, once obtained, is reused (design 10.3 and 11)."""

import hashlib
import json
import sqlite3
from typing import Any


def cache_key(
    slug: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    params: dict[str, Any],
    prompt_version: str,
) -> str:
    """Hash of everything that decides a response. `messages` carry image digests."""
    material = [slug, messages, schema, params, prompt_version]
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class CallCache:
    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        db.execute(
            "CREATE TABLE IF NOT EXISTS calls"
            " (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )

    def get(self, key: str) -> str | None:
        """The raw response JSON stored under `key`, if any."""
        row = self._db.execute("SELECT response FROM calls WHERE key = ?", (key,))
        found: tuple[str] | None = row.fetchone()
        return found[0] if found else None

    def put(self, key: str, response: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, response))
