"""The usage ledger: one row per network call, kept in the call cache's SQLite file."""

import sqlite3
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost: Decimal
    """USD."""

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost + other.cost,
        )


class Ledger:
    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        db.execute(
            "CREATE TABLE IF NOT EXISTS ledger (role TEXT NOT NULL, slug TEXT NOT NULL,"
            " prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,"
            " cost TEXT NOT NULL)"
        )

    def record(self, role: str, slug: str, usage: Usage) -> None:
        self._db.execute(
            "INSERT INTO ledger VALUES (?, ?, ?, ?, ?)",
            (role, slug, usage.prompt_tokens, usage.completion_tokens, str(usage.cost)),
        )

    def totals_by_role(self) -> dict[str, Usage]:
        totals: dict[str, Usage] = {}
        rows = self._db.execute(
            "SELECT role, prompt_tokens, completion_tokens, cost FROM ledger"
        )
        for role, prompt_tokens, completion_tokens, cost in rows:
            usage = Usage(prompt_tokens, completion_tokens, Decimal(cost))
            totals[role] = totals[role] + usage if role in totals else usage
        return totals
