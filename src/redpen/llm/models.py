"""What the gateway needs to know about a model. Task 1B fills it from the API."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    """Concrete OpenRouter slug, such as `google/gemini-3.1-flash-lite`."""
    supported_parameters: frozenset[str]
    accepts_images: bool
    prompt_price: Decimal
    """USD per prompt token."""
    completion_price: Decimal
    """USD per completion token."""
