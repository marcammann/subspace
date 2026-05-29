from subspace.models.events import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)


def offset_output_index(event: StreamEvent, offset: int) -> StreamEvent:
    if offset == 0:
        return event

    if isinstance(
        event,
        (
            ResponseOutputItemAddedEvent,
            ResponseOutputItemDoneEvent,
            ResponseContentPartAddedEvent,
            ResponseContentPartDoneEvent,
            ResponseOutputTextDeltaEvent,
            ResponseOutputTextDoneEvent,
            ResponseFunctionCallArgsDeltaEvent,
            ResponseFunctionCallArgsDoneEvent,
        ),
    ):
        return event.model_copy(update={"output_index": event.output_index + offset})

    return event
