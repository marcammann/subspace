from typing import Literal

from pydantic import BaseModel

from subspace.models.content import OutputContent
from subspace.models.items import OutputItem
from subspace.models.response import ResponseResource


class BaseStreamEvent(BaseModel):
    type: str
    sequence_number: int


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"] = "response.created"
    response: ResponseResource


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponseResource


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"] = "response.completed"
    response: ResponseResource


class ResponseFailedEvent(BaseStreamEvent):
    type: Literal["response.failed"] = "response.failed"
    response: ResponseResource


class ResponseIncompleteEvent(BaseStreamEvent):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: ResponseResource


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: OutputItem


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: OutputItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int
    content_index: int
    part: OutputContent


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int
    content_index: int
    part: OutputContent


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseFunctionCallArgsDeltaEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgsDoneEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    item_id: str
    output_index: int
    call_id: str
    arguments: str


class ErrorEvent(BaseStreamEvent):
    type: Literal["error"] = "error"
    code: str
    message: str


type BuiltInStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseFunctionCallArgsDeltaEvent
    | ResponseFunctionCallArgsDoneEvent
    | ErrorEvent
)
type TerminalStreamEvent = ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent

type StreamEvent = BaseStreamEvent
