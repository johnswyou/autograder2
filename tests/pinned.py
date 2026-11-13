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
# ~anthropic/claude-opus-latest as well
REFEREE = SOLVER
# ~openai/gpt-sol-latest, as resolved on 13 November 2026
CHECKER = ModelInfo(
    slug="openai/gpt-6.1-sol",
    supported_parameters=frozenset(
        {
            "include_reasoning",
            "max_completion_tokens",
            "max_tokens",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "seed",
            "structured_outputs",
            "tool_choice",
            "tools",
            "verbosity",
        }
    ),
    accepts_images=True,
    prompt_price=Decimal("0.000002"),
    completion_price=Decimal("0.00001"),
    providers=frozenset({"azure", "azure/eu", "azure/us"}),
)
# The roles `redpen prepare` and `redpen key verify` call once the spec exists: the
# grader tries the rubric.
KEY_MODELS = {
    "solver": SOLVER,
    "checker": CHECKER,
    "referee": REFEREE,
    "grader": GRADER,
}
# ~anthropic/claude-haiku-latest
MAPPER = ModelInfo(
    slug="anthropic/claude-haiku-5.5",
    supported_parameters=_PARAMETERS,
    accepts_images=True,
    prompt_price=Decimal("1E-7"),
    completion_price=Decimal("5E-7"),
    providers=_VERTEX,
)
# ~openai/gpt-sol-latest as well: the second reader, of another vendor than the
# reader's
SECOND = CHECKER
# The roles `redpen grade` sends student pages to.
GRADE_MODELS = {"mapper": MAPPER, "reader": READER, "second": SECOND, "grader": GRADER}
# OpenRouter's ZDR listing as it held those models' pinned endpoints: none retains.
ZDR = [
    {
        "model_id": slug,
        "tag": tag,
        "safety_retention": False,
        "supported_parameters": ["structured_outputs"],
    }
    for slug, model in sorted({m.slug: m for m in GRADE_MODELS.values()}.items())
    for tag in sorted(model.providers)
]
