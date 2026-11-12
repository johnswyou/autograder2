"""The models the recordings were made with: each role's default alias as `pin`
resolved it with strict privacy on, 12 November 2026. A recorded call's cache key
holds the slug and the parameters sent, so a replay must pass the same model."""

from decimal import Decimal

from redpen.llm.models import ModelInfo

_PARAMETERS = frozenset(
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
)
_VERTEX = frozenset(
    {"google-vertex/europe", "google-vertex/global", "google-vertex/us"}
)

# ~anthropic/claude-sonnet-latest
PARSER = ModelInfo(
    slug="anthropic/claude-sonnet-5.5",
    supported_parameters=_PARAMETERS,
    accepts_images=True,
    prompt_price=Decimal("0.000002"),
    completion_price=Decimal("0.00001"),
    providers=_VERTEX,
)
# ~anthropic/claude-sonnet-latest as well
READER = PARSER
GRADER = PARSER
# ~anthropic/claude-opus-latest
SOLVER = ModelInfo(
    slug="anthropic/claude-opus-5.5",
    supported_parameters=_PARAMETERS,
    accepts_images=True,
    prompt_price=Decimal("0.000004"),
    completion_price=Decimal("0.00002"),
    providers=_VERTEX,
)
# ~anthropic/claude-haiku-latest
MAPPER = ModelInfo(
    slug="anthropic/claude-haiku-5.5",
    supported_parameters=_PARAMETERS,
    accepts_images=True,
    prompt_price=Decimal("1E-7"),
    completion_price=Decimal("5E-7"),
    providers=_VERTEX,
)
