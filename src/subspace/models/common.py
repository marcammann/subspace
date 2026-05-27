from enum import StrEnum

from pydantic import BaseModel


class Status(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    QUEUED = "queued"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"


class ToolChoiceEnum(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class TruncationEnum(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails | None = None
    output_tokens_details: OutputTokensDetails | None = None


class ResponseError(BaseModel):
    message: str
    type: str
    code: str | None = None
    param: str | None = None
