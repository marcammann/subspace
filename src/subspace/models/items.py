from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from subspace.models.common import Role, Status
from subspace.models.content import InputContent, OutputContent


class BaseItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str  # subclasses narrow this to a Literal


class InputMessage(BaseItem):
    type: Literal["message"] = "message"
    role: Role
    content: str | list[InputContent]
    status: Status | None = None


class OutputMessage(BaseItem):
    type: Literal["message"] = "message"
    id: str
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: list[OutputContent] = Field(default_factory=list)
    status: Status = Status.IN_PROGRESS


class FunctionCall(BaseItem):
    type: Literal["function_call"] = "function_call"
    id: str
    name: str
    call_id: str
    arguments: str
    status: Status = Status.IN_PROGRESS


class FunctionCallOutput(BaseItem):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class ServerFunctionCall(BaseItem):
    """A tool call that was executed server-side by a middleware.

    Unlike FunctionCall (which signals the client should execute the tool),
    this represents a completed call+result pair. Interfaces can render it
    however they choose (e.g. as a specialized item type, or suppress it).
    """

    type: Literal["server_function_call"] = "server_function_call"
    id: str
    name: str
    call_id: str
    arguments: str
    output: str
    status: Status = Status.COMPLETED


InputItem = Annotated[
    InputMessage | FunctionCallOutput,
    Field(discriminator="type"),
]

OutputItem = Annotated[
    OutputMessage | FunctionCall | ServerFunctionCall,
    Field(discriminator="type"),
]

# Broad union for round-trip inputs (includes output items fed back as context)
AnyItem = InputMessage | OutputMessage | FunctionCall | FunctionCallOutput | ServerFunctionCall
