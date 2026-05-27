"""Shared state machine for building OpenResponses streaming events.

Used by litellm, echo, and langchain backends to emit a consistent
event sequence without duplicating the text/tool-call bookkeeping.
"""

import uuid
from collections.abc import Iterator

from subspace.models.common import ResponseError, Status, Usage
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage
from subspace.models.response import ResponseResource


class ResponseBuilder:
    """Translates streaming chunks into the OpenResponses event sequence.

    Manages sequence numbering, text accumulation, tool-call accumulation,
    and response lifecycle events (created → in_progress → completed/failed).
    """

    def __init__(self, response: ResponseResource) -> None:
        self._response = response
        self._seq = 0
        self._output_items: list[OutputMessage | FunctionCall] = []
        self._text_msg_id: str | None = None
        self._accumulated_text = ""
        self._text_started = False
        self._tool_calls: dict[int, _ToolCallState] = {}
        self._usage = Usage()

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def start(self) -> Iterator[StreamEvent]:
        yield ResponseCreatedEvent(sequence_number=self._next_seq(), response=self._response)
        yield ResponseInProgressEvent(sequence_number=self._next_seq(), response=self._response)

    def text_delta(self, content: str) -> Iterator[StreamEvent]:
        if not self._text_started:
            self._text_msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            yield ResponseOutputItemAddedEvent(
                sequence_number=self._next_seq(),
                output_index=len(self._output_items),
                item=OutputMessage(id=self._text_msg_id, content=[], status=Status.IN_PROGRESS),
            )
            yield ResponseContentPartAddedEvent(
                sequence_number=self._next_seq(),
                item_id=self._text_msg_id,
                output_index=len(self._output_items),
                content_index=0,
                part=OutputTextContent(text=""),
            )
            self._text_started = True

        self._accumulated_text += content
        assert self._text_msg_id is not None
        yield ResponseOutputTextDeltaEvent(
            sequence_number=self._next_seq(),
            item_id=self._text_msg_id,
            output_index=len(self._output_items),
            content_index=0,
            delta=content,
        )

    def tool_call_delta(
        self,
        *,
        index: int = 0,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> Iterator[StreamEvent]:
        if index not in self._tool_calls:
            self._tool_calls[index] = _ToolCallState(
                call_id=call_id or f"call_{uuid.uuid4().hex[:24]}",
                name=name or "",
            )
        state = self._tool_calls[index]
        if call_id:
            state.call_id = call_id
        if name:
            state.name = name
        if not arguments:
            return

        state.arguments += arguments

        if not state.started:
            state.started = True
            yield from self._finalize_text()
            state.item_id = f"fc_{uuid.uuid4().hex[:24]}"
            state.output_index = len(self._output_items)
            yield ResponseOutputItemAddedEvent(
                sequence_number=self._next_seq(),
                output_index=state.output_index,
                item=FunctionCall(
                    id=state.item_id,
                    name=state.name,
                    call_id=state.call_id,
                    arguments="",
                    status=Status.IN_PROGRESS,
                ),
            )

        yield ResponseFunctionCallArgsDeltaEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            call_id=state.call_id,
            delta=arguments,
        )

    def call_ids(self) -> list[str]:
        return [s.call_id for _, s in sorted(self._tool_calls.items()) if s.started]

    def set_usage(self, usage: Usage) -> None:
        self._usage = usage

    def finish(self) -> Iterator[StreamEvent]:
        yield from self._finalize_text()
        yield from self._finalize_tool_calls()

        response = self._response.model_copy(
            update={
                "status": Status.COMPLETED,
                "output": self._output_items,
                "usage": self._usage,
            }
        )
        yield ResponseCompletedEvent(sequence_number=self._next_seq(), response=response)

    def fail(self, exc: Exception) -> Iterator[StreamEvent]:
        response = self._response.model_copy(
            update={
                "status": Status.FAILED,
                "error": ResponseError(
                    message=str(exc),
                    type="server_error",
                    code="server_error",
                ),
            }
        )
        yield ResponseFailedEvent(sequence_number=self._next_seq(), response=response)

    def _finalize_text(self) -> Iterator[StreamEvent]:
        if not self._text_started or not self._text_msg_id:
            return

        yield ResponseOutputTextDoneEvent(
            sequence_number=self._next_seq(),
            item_id=self._text_msg_id,
            output_index=len(self._output_items),
            content_index=0,
            text=self._accumulated_text,
        )

        done_part = OutputTextContent(text=self._accumulated_text)
        yield ResponseContentPartDoneEvent(
            sequence_number=self._next_seq(),
            item_id=self._text_msg_id,
            output_index=len(self._output_items),
            content_index=0,
            part=done_part,
        )

        done_msg = OutputMessage(
            id=self._text_msg_id,
            content=[done_part],
            status=Status.COMPLETED,
        )
        yield ResponseOutputItemDoneEvent(
            sequence_number=self._next_seq(),
            output_index=len(self._output_items),
            item=done_msg,
        )
        self._output_items.append(done_msg)
        self._text_started = False

    def _finalize_tool_calls(self) -> Iterator[StreamEvent]:
        for state in self._tool_calls.values():
            if not state.started:
                continue
            yield ResponseFunctionCallArgsDoneEvent(
                sequence_number=self._next_seq(),
                item_id=state.item_id,
                output_index=state.output_index,
                call_id=state.call_id,
                arguments=state.arguments,
            )
            fc_done = FunctionCall(
                id=state.item_id,
                name=state.name,
                call_id=state.call_id,
                arguments=state.arguments,
                status=Status.COMPLETED,
            )
            yield ResponseOutputItemDoneEvent(
                sequence_number=self._next_seq(),
                output_index=state.output_index,
                item=fc_done,
            )
            self._output_items.append(fc_done)


class _ToolCallState:
    __slots__ = ("call_id", "name", "arguments", "started", "item_id", "output_index")

    def __init__(self, call_id: str, name: str) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = ""
        self.started = False
        self.item_id = ""
        self.output_index = 0
