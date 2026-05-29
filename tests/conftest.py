import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from subspace.middleware.context import RequestContext
from subspace.models.common import ResponseError, Status, Usage
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource
from subspace.models.tools import FunctionTool


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_ctx(
    input: str | list = "hello",
    instructions: str | None = None,
    tools: list[FunctionTool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RequestContext:
    request = CreateResponseRequest(
        model="test", input=input, instructions=instructions, tools=tools
    )
    response = ResponseResource(
        id="resp_test",
        created_at=int(time.time()),
        status=Status.IN_PROGRESS,
        model="test",
    )
    return RequestContext(
        request=request,
        response_id="resp_test",
        response=response,
        metadata=metadata or {},
    )


def make_response(
    status: Status = Status.IN_PROGRESS,
    model: str = "test",
) -> ResponseResource:
    return ResponseResource(
        id="resp_test",
        created_at=int(time.time()),
        status=status,
        model=model,
    )


def text_backend_events(text: str = "Hello world") -> list[StreamEvent]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0
    response = make_response()
    events: list[StreamEvent] = []

    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(
        ResponseOutputItemAddedEvent(
            sequence_number=seq,
            output_index=0,
            item=OutputMessage(id=msg_id, content=[], status=Status.IN_PROGRESS),
        )
    )
    seq += 1
    events.append(
        ResponseContentPartAddedEvent(
            sequence_number=seq,
            item_id=msg_id,
            output_index=0,
            content_index=0,
            part=OutputTextContent(text=""),
        )
    )
    seq += 1

    for word in text.split():
        events.append(
            ResponseOutputTextDeltaEvent(
                sequence_number=seq,
                item_id=msg_id,
                output_index=0,
                content_index=0,
                delta=word + " ",
            )
        )
        seq += 1

    events.append(
        ResponseOutputTextDoneEvent(
            sequence_number=seq,
            item_id=msg_id,
            output_index=0,
            content_index=0,
            text=text,
        )
    )
    seq += 1
    events.append(
        ResponseContentPartDoneEvent(
            sequence_number=seq,
            item_id=msg_id,
            output_index=0,
            content_index=0,
            part=OutputTextContent(text=text),
        )
    )
    seq += 1

    done_msg = OutputMessage(
        id=msg_id,
        content=[OutputTextContent(text=text)],
        status=Status.COMPLETED,
    )
    events.append(
        ResponseOutputItemDoneEvent(sequence_number=seq, output_index=0, item=done_msg)
    )
    seq += 1

    completed_response = response.model_copy(
        update={
            "status": Status.COMPLETED,
            "output": [done_msg],
            "usage": Usage(input_tokens=2, output_tokens=2, total_tokens=4),
        }
    )
    events.append(ResponseCompletedEvent(sequence_number=seq, response=completed_response))
    return events


def error_backend_events() -> list[StreamEvent]:
    seq = 0
    response = make_response()
    events: list[StreamEvent] = []
    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1
    failed_response = response.model_copy(
        update={
            "status": Status.FAILED,
            "error": ResponseError(message="boom", type="server_error", code="server_error"),
        }
    )
    events.append(ResponseFailedEvent(sequence_number=seq, response=failed_response))
    return events


def function_call_backend_events(
    name: str = "get_weather",
    call_id: str = "call_abc",
    arguments: str = '{"city": "London"}',
) -> list[StreamEvent]:
    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
    seq = 0
    response = make_response()
    events: list[StreamEvent] = []

    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1

    fc = FunctionCall(
        id=fc_id, name=name, call_id=call_id, arguments="", status=Status.IN_PROGRESS
    )
    events.append(
        ResponseOutputItemAddedEvent(sequence_number=seq, output_index=0, item=fc)
    )
    seq += 1

    for chunk in [arguments[:len(arguments) // 2], arguments[len(arguments) // 2:]]:
        events.append(
            ResponseFunctionCallArgsDeltaEvent(
                sequence_number=seq,
                item_id=fc_id,
                output_index=0,
                call_id=call_id,
                delta=chunk,
            )
        )
        seq += 1

    events.append(
        ResponseOutputItemDoneEvent(
            sequence_number=seq,
            output_index=0,
            item=FunctionCall(
                id=fc_id,
                name=name,
                call_id=call_id,
                arguments=arguments,
                status=Status.COMPLETED,
            ),
        )
    )
    seq += 1

    completed_response = response.model_copy(
        update={
            "status": Status.COMPLETED,
            "output": [fc],
            "usage": Usage(input_tokens=5, output_tokens=3, total_tokens=8),
        }
    )
    events.append(ResponseCompletedEvent(sequence_number=seq, response=completed_response))
    return events


class StaticBackend:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event
