"""Builds a Subspace-compatible response stream from backend output.

The stream format is inspired by the OpenResponses specification.
Backends use this to translate their native streaming into Subspace's
internal event model without duplicating the bookkeeping.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

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
from subspace.models.items import FunctionCall, OutputMessage, ServerFunctionCall
from subspace.models.response import ResponseResource


class ResponseBuilder:
    """Translates backend output into a Subspace response stream.

    Manages sequence numbering, text accumulation, tool-call accumulation,
    and response lifecycle events (created → in_progress → completed/failed).
    """

    def __init__(self, response: ResponseResource) -> None:
        self._response = response
        self._seq = 0
        self._next_output_index = 0
        self._phase = _BuilderPhase.NEW
        self._output_items: list[OutputMessage | FunctionCall | ServerFunctionCall] = []
        self._open_text: _TextItemState | None = None
        self._tool_calls_by_index: dict[int, _ToolCallState] = {}
        self._tool_calls_by_call_id: dict[str, _ToolCallState] = {}
        self._pending_server_outputs: dict[str, str] = {}
        self._usage = Usage()

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _allocate_output_index(self) -> int:
        output_index = self._next_output_index
        self._next_output_index += 1
        return output_index

    def start(self) -> Iterator[StreamEvent]:
        """Emit the response.created and response.in_progress lifecycle events."""
        self._require_phase(_BuilderPhase.NEW)
        self._phase = _BuilderPhase.STARTED
        yield ResponseCreatedEvent(sequence_number=self._next_seq(), response=self._response)
        yield ResponseInProgressEvent(sequence_number=self._next_seq(), response=self._response)

    def text_delta(self, content: str) -> Iterator[StreamEvent]:
        """Append streamed assistant text, opening a message item when needed."""
        self._require_started()
        if self._open_text is None:
            yield from self._open_text_item()

        assert self._open_text is not None
        self._open_text.text += content
        yield ResponseOutputTextDeltaEvent(
            sequence_number=self._next_seq(),
            item_id=self._open_text.item_id,
            output_index=self._open_text.output_index,
            content_index=self._open_text.content_index,
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
        """Append a streamed function-call chunk for the provider tool index."""
        self._require_started()
        state = self._get_or_create_tool_state(index, call_id=call_id, name=name)
        if call_id and call_id != state.call_id:
            self._tool_calls_by_call_id.pop(state.call_id, None)
            state.call_id = call_id
            self._tool_calls_by_call_id[state.call_id] = state
        if name:
            state.name = name

        if state.phase is _ItemPhase.CLOSED:
            msg = f"Tool call {state.call_id!r} is already finalized"
            raise RuntimeError(msg)

        if not state.opened and arguments:
            state.arguments += arguments

        if not state.opened and state.name:
            yield from self._open_tool_call(state)
            if state.arguments:
                yield ResponseFunctionCallArgsDeltaEvent(
                    sequence_number=self._next_seq(),
                    item_id=state.item_id,
                    output_index=state.output_index,
                    call_id=state.call_id,
                    delta=state.arguments,
                )
        elif not state.opened:
            return

        elif arguments:
            state.arguments += arguments
            yield ResponseFunctionCallArgsDeltaEvent(
                sequence_number=self._next_seq(),
                item_id=state.item_id,
                output_index=state.output_index,
                call_id=state.call_id,
                delta=arguments,
            )

        if state.call_id in self._pending_server_outputs:
            output = self._pending_server_outputs.pop(state.call_id)
            yield from self._finalize_server_tool_call(state, output)

    def call_ids(self) -> list[str]:
        """Return opened function call IDs sorted by provider tool index."""
        return [
            state.call_id for _, state in sorted(self._tool_calls_by_index.items()) if state.opened
        ]

    def server_tool_output(self, call_id: str, output: str) -> Iterator[StreamEvent]:
        """Finalize a function call as a server-executed tool result."""
        self._require_started()
        state = self._tool_calls_by_call_id.get(call_id)
        if state is None:
            self._pending_server_outputs[call_id] = (
                self._pending_server_outputs.get(call_id, "") + output
            )
            return

        if state.phase is _ItemPhase.CLOSED:
            msg = f"Tool call {call_id!r} is already finalized"
            raise RuntimeError(msg)

        if not state.opened:
            yield from self._open_tool_call(state)
        yield from self._finalize_server_tool_call(state, output)

    def set_usage(self, usage: Usage) -> None:
        """Set token usage to include on the completed response."""
        self._usage = usage

    def finish(self) -> Iterator[StreamEvent]:
        """Close open items and emit the terminal response.completed event."""
        self._require_started()
        yield from self._finalize_text()
        yield from self._finalize_tool_calls()

        response = self._response.model_copy(
            update={
                "status": Status.COMPLETED,
                "output": self._output_items,
                "usage": self._usage,
            }
        )
        self._phase = _BuilderPhase.COMPLETED
        yield ResponseCompletedEvent(sequence_number=self._next_seq(), response=response)

    def fail(self, exc: Exception) -> Iterator[StreamEvent]:
        """Emit the terminal response.failed event for a backend exception."""
        self._require_started()
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
        self._phase = _BuilderPhase.FAILED
        yield ResponseFailedEvent(sequence_number=self._next_seq(), response=response)

    def _require_phase(self, phase: "_BuilderPhase") -> None:
        if self._phase is not phase:
            msg = f"ResponseBuilder expected phase {phase}, got {self._phase}"
            raise RuntimeError(msg)

    def _require_started(self) -> None:
        if self._phase is _BuilderPhase.STARTED:
            return
        if self._phase in (_BuilderPhase.COMPLETED, _BuilderPhase.FAILED):
            msg = "ResponseBuilder is already terminal"
        else:
            msg = "ResponseBuilder.start() must be called before streaming events"
        raise RuntimeError(msg)

    def _open_text_item(self) -> Iterator[StreamEvent]:
        state = _TextItemState(
            item_id=f"msg_{uuid.uuid4().hex[:24]}",
            output_index=self._allocate_output_index(),
        )
        self._open_text = state
        yield ResponseOutputItemAddedEvent(
            sequence_number=self._next_seq(),
            output_index=state.output_index,
            item=OutputMessage(id=state.item_id, content=[], status=Status.IN_PROGRESS),
        )
        yield ResponseContentPartAddedEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=OutputTextContent(text=""),
        )

    def _finalize_text(self) -> Iterator[StreamEvent]:
        state = self._open_text
        if state is None:
            return

        yield ResponseOutputTextDoneEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            text=state.text,
        )

        done_part = OutputTextContent(text=state.text)
        yield ResponseContentPartDoneEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=done_part,
        )

        done_msg = OutputMessage(
            id=state.item_id,
            content=[done_part],
            status=Status.COMPLETED,
        )
        yield ResponseOutputItemDoneEvent(
            sequence_number=self._next_seq(),
            output_index=state.output_index,
            item=done_msg,
        )
        self._output_items.append(done_msg)
        state.phase = _ItemPhase.CLOSED
        self._open_text = None

    def _finalize_tool_calls(self) -> Iterator[StreamEvent]:
        for state in sorted(self._tool_calls_by_index.values(), key=lambda s: s.output_index):
            if not state.opened or state.phase is _ItemPhase.CLOSED:
                continue
            yield from self._finalize_client_tool_call(state)

    def _get_or_create_tool_state(
        self,
        index: int,
        *,
        call_id: str | None,
        name: str | None,
    ) -> "_ToolCallState":
        if index not in self._tool_calls_by_index:
            state = _ToolCallState(
                provider_index=index,
                call_id=call_id or f"call_{uuid.uuid4().hex[:24]}",
                name=name or "",
            )
            self._tool_calls_by_index[index] = state
            self._tool_calls_by_call_id[state.call_id] = state
        return self._tool_calls_by_index[index]

    def _open_tool_call(self, state: "_ToolCallState") -> Iterator[StreamEvent]:
        yield from self._finalize_text()
        state.opened = True
        state.item_id = f"fc_{uuid.uuid4().hex[:24]}"
        state.output_index = self._allocate_output_index()
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

    def _finalize_client_tool_call(self, state: "_ToolCallState") -> Iterator[StreamEvent]:
        yield ResponseFunctionCallArgsDoneEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            call_id=state.call_id,
            arguments=state.arguments,
        )
        item = FunctionCall(
            id=state.item_id,
            name=state.name,
            call_id=state.call_id,
            arguments=state.arguments,
            status=Status.COMPLETED,
        )
        yield ResponseOutputItemDoneEvent(
            sequence_number=self._next_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)
        state.phase = _ItemPhase.CLOSED

    def _finalize_server_tool_call(
        self, state: "_ToolCallState", output: str
    ) -> Iterator[StreamEvent]:
        yield ResponseFunctionCallArgsDoneEvent(
            sequence_number=self._next_seq(),
            item_id=state.item_id,
            output_index=state.output_index,
            call_id=state.call_id,
            arguments=state.arguments,
        )
        item = ServerFunctionCall(
            id=state.item_id,
            name=state.name,
            call_id=state.call_id,
            arguments=state.arguments,
            output=output,
        )
        yield ResponseOutputItemDoneEvent(
            sequence_number=self._next_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)
        state.phase = _ItemPhase.CLOSED


class _BuilderPhase(StrEnum):
    NEW = "new"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class _ItemPhase(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class _TextItemState:
    item_id: str
    output_index: int
    content_index: int = 0
    text: str = ""
    phase: _ItemPhase = _ItemPhase.OPEN


@dataclass
class _ToolCallState:
    provider_index: int
    call_id: str
    name: str
    arguments: str = ""
    opened: bool = False
    item_id: str = ""
    output_index: int = 0
    phase: _ItemPhase = _ItemPhase.OPEN
