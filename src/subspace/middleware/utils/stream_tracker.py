"""Tracks open output items in a response stream.

Feed every event to `track()`. When you need to cancel the stream early,
call `close_open_items()` to get the events that properly close any
in-progress text or tool-call items.

    tracker = StreamTracker()
    async for event in upstream:
        tracker.track(event)
        if should_cancel:
            for close_event in tracker.close_open_items():
                yield close_event
            break
        yield event
"""

from dataclasses import dataclass, field

from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    BaseStreamEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage
from subspace.models.response import ResponseResource


@dataclass
class _OpenMessage:
    item_id: str
    output_index: int
    content_index: int = 0
    text: str = ""


@dataclass
class _OpenFunctionCall:
    item_id: str
    output_index: int
    call_id: str
    name: str
    arguments: str = ""


@dataclass
class StreamTracker:
    """Tracks stream state so open items can be closed on early cancellation."""

    response: ResponseResource | None = field(default=None, init=False)
    completed_items: list = field(default_factory=list, init=False)
    last_text_item_id: str | None = field(default=None, init=False)
    last_text_output_index: int = field(default=0, init=False)

    _seq: int = field(default=0, init=False)
    _open_item: _OpenMessage | _OpenFunctionCall | None = field(default=None, init=False)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def track(self, event: StreamEvent) -> None:
        if isinstance(event, BaseStreamEvent):
            self._seq = event.sequence_number

        if isinstance(event, ResponseCreatedEvent):
            self.response = event.response

        elif isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, OutputMessage):
                self._open_item = _OpenMessage(
                    item_id=event.item.id, output_index=event.output_index
                )
                self.last_text_item_id = event.item.id
                self.last_text_output_index = event.output_index
            elif isinstance(event.item, FunctionCall):
                self._open_item = _OpenFunctionCall(
                    item_id=event.item.id,
                    output_index=event.output_index,
                    call_id=event.item.call_id,
                    name=event.item.name,
                )

        elif isinstance(event, ResponseOutputTextDeltaEvent):
            if isinstance(self._open_item, _OpenMessage):
                self._open_item.text += event.delta
                self._open_item.content_index = event.content_index

        elif isinstance(event, ResponseFunctionCallArgsDeltaEvent):
            if isinstance(self._open_item, _OpenFunctionCall):
                self._open_item.arguments += event.delta

        elif isinstance(event, ResponseOutputItemDoneEvent):
            self.completed_items.append(event.item)
            self._open_item = None

    def close_open_items(self) -> list[StreamEvent]:
        """Emit done events for the current open item, if any."""
        if self._open_item is None:
            return []

        events: list[StreamEvent] = []

        if isinstance(self._open_item, _OpenMessage):
            msg = self._open_item
            events.append(
                ResponseOutputTextDoneEvent(
                    sequence_number=self.next_seq(),
                    item_id=msg.item_id,
                    output_index=msg.output_index,
                    content_index=msg.content_index,
                    text=msg.text,
                )
            )
            done_part = OutputTextContent(text=msg.text)
            events.append(
                ResponseContentPartDoneEvent(
                    sequence_number=self.next_seq(),
                    item_id=msg.item_id,
                    output_index=msg.output_index,
                    content_index=msg.content_index,
                    part=done_part,
                )
            )
            done_item = OutputMessage(
                id=msg.item_id,
                content=[done_part],
                status=Status.INCOMPLETE,
            )
            events.append(
                ResponseOutputItemDoneEvent(
                    sequence_number=self.next_seq(),
                    output_index=msg.output_index,
                    item=done_item,
                )
            )
            self.completed_items.append(done_item)

        elif isinstance(self._open_item, _OpenFunctionCall):
            fc = self._open_item
            events.append(
                ResponseFunctionCallArgsDoneEvent(
                    sequence_number=self.next_seq(),
                    item_id=fc.item_id,
                    output_index=fc.output_index,
                    call_id=fc.call_id,
                    arguments=fc.arguments,
                )
            )
            done_item = FunctionCall(
                id=fc.item_id,
                name=fc.name,
                call_id=fc.call_id,
                arguments=fc.arguments,
                status=Status.COMPLETED,
            )
            events.append(
                ResponseOutputItemDoneEvent(
                    sequence_number=self.next_seq(),
                    output_index=fc.output_index,
                    item=done_item,
                )
            )
            self.completed_items.append(done_item)

        self._open_item = None
        return events

