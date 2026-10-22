"""API contracts"""

from pydantic import BaseModel, Field
from enum import Enum


class Model(Enum):
    SMART = "smart"
    MEDIUM = "medium"
    LIGHT = "light"


class Request(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: Model = Field(Model.SMART)
