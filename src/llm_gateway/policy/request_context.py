"""Request context for policy pipeline."""

from dataclasses import dataclass, field
from enum import Enum
import time


class OperationType(str, Enum):
    GENERATE = "generate"
    STREAM = "stream"
    EXTRACT = "extract"

@dataclass(slots=True)
class RequestContext:
    request_id: str
    operation: OperationType
    prompt: str
    deadline: float

    primary_attempts: int = 0
    fallback_attempts: int = 0

    primary_model: str | None = None
    current_model: str | None = None

    first_chunk_emitted: bool = False
    attempted_models: set[str] = field(default_factory=set)
    model_errors: dict[str, Exception] = field(default_factory=dict)

    @property
    def remaining_time(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def is_expired(self) -> bool:
        return self.remaining_time <= 0.0
