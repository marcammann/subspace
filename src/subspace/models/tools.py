from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=64)
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    name: str


ToolChoice = str | ToolChoiceFunction
Tool = FunctionTool
