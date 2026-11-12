"""A recorded call to a real model, replayed with the network disabled.

Re-record (paid) with `REDPEN_RECORD=1 uv run pytest tests/replay`.
"""

import asyncio
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from redpen.llm.client import Gateway, Result
from redpen.llm.models import ModelInfo

IMAGE = (Path(__file__).parents[1] / "fixtures" / "letter_b.png").read_bytes()

# ~anthropic/claude-haiku-latest as `pin` resolved it with strict privacy on
# 12 November 2026.
MODEL = ModelInfo(
    slug="anthropic/claude-haiku-5.5",
    supported_parameters=frozenset(
        {
            "include_reasoning",
            "max_completion_tokens",
            "max_tokens",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "stop",
            "structured_outputs",
            "tool_choice",
            "tools",
            "verbosity",
        }
    ),
    accepts_images=True,
    prompt_price=Decimal("1E-7"),
    completion_price=Decimal("5E-7"),
    providers=frozenset(
        {"google-vertex/europe", "google-vertex/global", "google-vertex/us"}
    ),
)


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str


def test_a_recorded_call_replays_offline(replay: Callable[[str], Gateway]) -> None:
    async def main() -> Result[Letter]:
        async with replay("gateway_letter") as gateway:
            return await gateway.call(
                "mapper",
                MODEL,
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
    assert result.model_slug == MODEL.slug
