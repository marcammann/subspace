from typing import Any

from pydantic import BaseModel, Field, field_validator

from subspace.models.common import Role
from subspace.models.items import AnyItem, InputMessage
from subspace.models.tools import Tool, ToolChoice


class CreateResponseRequest(BaseModel):
    model: str
    input: list[AnyItem] = Field(default_factory=list)
    instructions: str | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=16)
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    stream: bool = True
    store: bool | None = None
    metadata: dict[str, str] | None = None
    user: str | None = Field(default=None, max_length=64)
    truncation: str | None = None
    extra: dict[str, Any] | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _normalize_input(cls, v: Any) -> list:
        if isinstance(v, str):
            return [InputMessage(role=Role.USER, content=v)] if v else []
        return v
