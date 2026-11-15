"""A recorded call to a real model, replayed with the network disabled.

Re-record (paid) with `AUTOGRADER2_RECORD=1 uv run pytest tests/replay`.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from autograder2.llm.client import Gateway, Result
from tests.pinned import MAPPER

IMAGE = (Path(__file__).parents[1] / "fixtures" / "letter_b.png").read_bytes()


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str


def test_a_recorded_call_replays_offline(replay: Callable[[str], Gateway]) -> None:
    async def main() -> Result[Letter]:
        async with replay("gateway_letter") as gateway:
            return await gateway.call(
                "mapper",
                MAPPER,
                "Give the capital letter shown in the image.",
                ["The image follows."],
                [IMAGE],
                Letter,
                "replay-1",
                params={"max_tokens": 300},
                session_id="replay:letter",
            )

    result = asyncio.run(main())

    assert result.value == Letter(letter="B")
    assert result.model_slug == MAPPER.slug
