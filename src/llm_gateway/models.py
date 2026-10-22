"""API contracts"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(str, Enum):
    SMART = "smart"
    MEDIUM = "medium"
    LIGHT = "light"


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=10000)
    model: Model = Field(Model.SMART)


StreamChunk = Annotated[
    str,
    Field(description="Один текстовый фрагмент потокового ответа."),
]


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    provider: Literal["openai", "claude", "ollama"]
    model: str = Field(min_length=1)


class ExtractedReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "negative", "mixed", "neutral"]
    topics: list[str]
    rating: int = Field(
        description="Оценка от 1 до 5, если известна; иначе 0.",
        ge=0,
        le=5,
    )
