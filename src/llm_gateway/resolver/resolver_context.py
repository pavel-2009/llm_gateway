"""Validated input context for model resolution."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_gateway.models import Model as ModelTier
from llm_gateway.policy.request_context import OperationType


class ResolveParams(BaseModel):
    """Optional parameters used to build the pipeline request context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    deadline: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class ResolveContext(BaseModel):
    """Validated values required to resolve a model request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: ModelTier = Field(strict=False)
    operation: OperationType
    prompt: str
    params: ResolveParams = Field(default_factory=ResolveParams)

    @field_validator("model", "prompt")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value
