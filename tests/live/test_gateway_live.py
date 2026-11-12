"""Paid calls through the real gateway. Run with `REDPEN_LIVE=1 uv run pytest -s
tests/live`."""

import asyncio
import io
import os
import sqlite3
from collections.abc import Awaitable
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from redpen.config import openrouter_api_key
from redpen.llm.client import Gateway, Result
from redpen.llm.ledger import Ledger
from redpen.llm.models import fetch_catalog
from redpen.schemas.grade import Grade

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("REDPEN_LIVE") != "1", reason="paid; set REDPEN_LIVE=1"
    ),
]

# The cheapest model found on 12 November 2026 with image input, structured outputs
# and an endpoint that retains nothing.
SLUG = "google/gemma-4-26b-a4b-it"
PARAMS = {"temperature": 0, "max_tokens": 300}


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str


def white_square_with_b() -> bytes:
    image = Image.new("RGB", (128, 128), "white")
    font = ImageFont.load_default(size=96)
    ImageDraw.Draw(image).text((64, 64), "B", fill="black", font=font, anchor="mm")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def ledger_rows(db: Path) -> int:
    with closing(sqlite3.connect(db)) as connection:
        return connection.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]


def test_live_gateway(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    image = white_square_with_b()

    async def main() -> None:
        catalog = await fetch_catalog(openrouter_api_key())
        model = catalog.resolve(SLUG, strict_privacy=True)
        print("\nproviders:", sorted(model.providers))
        async with Gateway(db, openrouter_api_key()) as gateway:

            def ask_letter() -> Awaitable[Result[Letter]]:
                return gateway.call(
                    "live",
                    model,
                    "Give the capital letter shown in the image.",
                    ["The image follows."],  # shared, so a cache breakpoint is sent
                    [image],
                    Letter,
                    "live-1",
                    params=PARAMS,
                    session_id="live:letter",
                )

            letter = await ask_letter()
            print("\nletter:", letter)
            assert letter.value.letter == "B"
            assert letter.model_slug == model.slug

            grade = await gateway.call(
                "live",
                model,
                "Grade part Q1 with one criterion, Q1.c1, worth 1 point. Award 1 if"
                " the answer is 42. Cite line 1 of the reading as evidence, no"
                " element. Leave the other fields null.",
                [],
                ["Reading of the student's response, line 1: 42"],
                Grade,
                "live-1",
                params=PARAMS,
            )
            print("grade:", grade)

            rows = ledger_rows(db)
            again = await ask_letter()
            print("again:", again)
            assert again.from_cache
            assert ledger_rows(db) == rows  # the cache hit wrote no row

    asyncio.run(main())
    with closing(sqlite3.connect(db)) as connection:
        print(
            "ledger rows:",
            ledger_rows(db),
            "totals:",
            Ledger(connection).totals_by_role(),
        )
