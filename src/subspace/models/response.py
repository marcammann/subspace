from typing import Any

from pydantic import BaseModel, Field

from subspace.models.common import ResponseError, Status, Usage
from subspace.models.items import InputItem, OutputItem
from subspace.models.tools import Tool, ToolChoice


class ResponseResource(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: Status = Status.IN_PROGRESS
    model: str
    input: list[InputItem] = Field(default_factory=list)
    output: list[OutputItem] = Field(default_factory=list)
    instructions: str | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    truncation: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    usage: Usage | None = None
    error: ResponseError | None = None
    metadata: dict[str, str] | None = None
    user: str | None = None
    store: bool | None = None
    incomplete_details: dict[str, Any] | None = None
