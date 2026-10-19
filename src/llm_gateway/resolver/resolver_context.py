"""Context for model resolver"""

from dataclasses import dataclass
from enum import Enum

from llm_gateway.policy.request_context import OperationType


@dataclass(slots=True)
class ResolveContext:
    model: str
    operation: OperationType
    prompt: str
    params: dict
