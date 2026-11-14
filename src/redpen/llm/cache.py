"""The call cache: a response, once obtained, is reused (design 10.3 and 11)."""

import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def image_digest(image: bytes) -> str:
    """SHA-256 of a PNG's decoded pixels (mode, size, palette, transparency and raw
    bytes), so that the same picture encoded again, as after a Pillow or zlib
    update, keeps its key. Any other format, or a PNG that does not decode, is
    keyed on its bytes."""
    if image.startswith(PNG_SIGNATURE):
        try:
            with Image.open(io.BytesIO(image)) as decoded:
                decoded.load()
                header = (
                    decoded.mode,
                    decoded.size,
                    decoded.getpalette(),
                    decoded.info.get("transparency"),
                )
                pixels = repr(header).encode() + decoded.tobytes()
        except Exception:  # an undecodable file is still keyed, on its bytes
            pass
        else:
            return hashlib.sha256(pixels).hexdigest()
    return hashlib.sha256(image).hexdigest()


def cache_key(
    slug: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    params: dict[str, Any],
    prompt_version: str,
    draw: int = 1,
) -> str:
    """Hash of everything that decides a response. `messages` carry image digests.
    A `draw` after the first keeps another answer to the same request apart; the
    first's key is the one kept before draws existed."""
    material: list[Any] = [slug, messages, schema, params, prompt_version]
    if draw != 1:
        material.append(draw)
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

    def forget(self, key: str) -> None:
        self._db.execute("DELETE FROM calls WHERE key = ?", (key,))

    def dump(self, path: Path) -> None:
        """Every row as a JSON line `{"key": ..., "response": {...}}`, in key order.

        Keys hold image digests, never images, so a dump holds responses only.
        """
        rows = self._db.execute("SELECT key, response FROM calls ORDER BY key")
        path.write_text(
            "".join(
                json.dumps({"key": key, "response": json.loads(response)}) + "\n"
                for key, response in rows
            )
        )

    def load(self, path: Path) -> None:
        """Add the rows of a file `dump` wrote."""
        for line in path.read_text().splitlines():
            row = json.loads(line)
            self.put(row["key"], json.dumps(row["response"]))
