import time

import pytest

from subspace.backends.response_builder import ResponseBuilder
from subspace.models.common import Status, Usage
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
)
from subspace.models.items import FunctionCall, OutputMessage, ServerFunctionCall
from subspace.models.response import ResponseResource


def _make_response() -> ResponseResource:
    return ResponseResource(
        id="resp_test",
        created_at=int(time.time()),
        status=Status.IN_PROGRESS,
        model="test",
    )


# ---------------------------------------------------------------------------
# start / finish lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_emits_created_and_in_progress(self):
        b = ResponseBuilder(_make_response())
        events = list(b.start())
        assert len(events) == 2
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], ResponseInProgressEvent)
        assert events[0].sequence_number == 0
        assert events[1].sequence_number == 1

    def test_finish_with_no_content(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(b.finish())
        assert len(events) == 1
        assert isinstance(events[0], ResponseCompletedEvent)
        assert events[0].response.status == Status.COMPLETED

    def test_fail_emits_failed(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(b.fail(RuntimeError("boom")))
        assert len(events) == 1
        assert isinstance(events[0], ResponseFailedEvent)
        assert events[0].response.status == Status.FAILED
        assert "boom" in events[0].response.error.message

    def test_set_usage_reflected_in_finish(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        b.set_usage(Usage(input_tokens=10, output_tokens=20, total_tokens=30))
        events = list(b.finish())
        assert events[-1].response.usage.input_tokens == 10
        assert events[-1].response.usage.output_tokens == 20


# ---------------------------------------------------------------------------
# text_delta
# ---------------------------------------------------------------------------


class TestTextDelta:
    def test_first_delta_emits_three_events(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(b.text_delta("hello"))
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[1], ResponseContentPartAddedEvent)
        assert isinstance(events[2], ResponseOutputTextDeltaEvent)
        assert events[2].delta == "hello"

    def test_second_delta_emits_one_event(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.text_delta("hello"))
        events = list(b.text_delta(" world"))
        assert len(events) == 1
        assert isinstance(events[0], ResponseOutputTextDeltaEvent)
        assert events[0].delta == " world"

    def test_finish_after_text_emits_done_events(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.text_delta("hello world"))
        events = list(b.finish())
        assert isinstance(events[0], ResponseOutputTextDoneEvent)
        assert events[0].text == "hello world"
        assert isinstance(events[1], ResponseContentPartDoneEvent)
        assert isinstance(events[2], ResponseOutputItemDoneEvent)
        assert isinstance(events[2].item, OutputMessage)
        assert events[2].item.status == Status.COMPLETED
        assert isinstance(events[3], ResponseCompletedEvent)

    def test_finish_noop_when_no_text(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(b.finish())
        assert len(events) == 1
        assert isinstance(events[0], ResponseCompletedEvent)

    def test_sequence_numbers_increment(self):
        b = ResponseBuilder(_make_response())
        list(b.start())  # 0, 1
        events = list(b.text_delta("hi"))
        assert events[0].sequence_number == 2
        assert events[1].sequence_number == 3
        assert events[2].sequence_number == 4


# ---------------------------------------------------------------------------
# tool_call_delta
# ---------------------------------------------------------------------------


class TestToolCallDelta:
    def test_first_chunk_emits_added_and_delta(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(
            b.tool_call_delta(index=0, call_id="call_123", name="my_tool", arguments='{"x":')
        )
        assert len(events) == 2
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[0].item, FunctionCall)
        assert events[0].item.name == "my_tool"
        assert isinstance(events[1], ResponseFunctionCallArgsDeltaEvent)
        assert events[1].delta == '{"x":'

    def test_accumulates_arguments(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments='{"a":'))
        events = list(b.tool_call_delta(index=0, arguments='"b"}'))
        assert len(events) == 1
        assert isinstance(events[0], ResponseFunctionCallArgsDeltaEvent)
        assert events[0].delta == '"b"}'

    def test_finish_emits_done(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments='{"a": 1}'))
        events = list(b.finish())
        assert isinstance(events[0], ResponseFunctionCallArgsDoneEvent)
        assert events[0].arguments == '{"a": 1}'
        assert isinstance(events[1], ResponseOutputItemDoneEvent)
        assert isinstance(events[1].item, FunctionCall)
        assert events[1].item.status == Status.COMPLETED
        assert isinstance(events[2], ResponseCompletedEvent)

    def test_multiple_tools(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_a", name="tool_a", arguments="{}"))
        list(b.tool_call_delta(index=1, call_id="call_b", name="tool_b", arguments="{}"))
        events = list(b.finish())
        # done + item_done for each tool, plus completed
        assert len(events) == 5

    def test_parallel_tools_get_distinct_output_indexes(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        first = list(b.tool_call_delta(index=0, call_id="call_a", name="tool_a", arguments="{}"))
        second = list(b.tool_call_delta(index=1, call_id="call_b", name="tool_b", arguments="{}"))

        assert first[0].output_index == 0
        assert second[0].output_index == 1

    def test_empty_arguments_still_starts_tool_call(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        events = list(b.tool_call_delta(index=0, call_id="call_1", name="t"))
        assert len(events) == 1
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[0].item, FunctionCall)
        assert events[0].item.call_id == "call_1"

        done = list(b.finish())
        assert isinstance(done[0], ResponseFunctionCallArgsDoneEvent)
        assert done[0].arguments == ""
        assert isinstance(done[1], ResponseOutputItemDoneEvent)
        assert isinstance(done[1].item, FunctionCall)
        assert done[1].item.arguments == ""

    def test_text_finalized_when_tool_call_starts(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.text_delta("thinking..."))
        events = list(b.tool_call_delta(index=0, call_id="c1", name="t", arguments="{}"))
        # text done, content part done, output item done, then tool added + delta
        assert isinstance(events[0], ResponseOutputTextDoneEvent)
        assert isinstance(events[1], ResponseContentPartDoneEvent)
        assert isinstance(events[2], ResponseOutputItemDoneEvent)
        assert isinstance(events[3], ResponseOutputItemAddedEvent)
        assert isinstance(events[4], ResponseFunctionCallArgsDeltaEvent)

    def test_text_tool_text_creates_separate_clean_messages(self):
        b = ResponseBuilder(_make_response())
        events = []
        events.extend(b.start())
        events.extend(b.text_delta("before"))
        events.extend(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments="{}"))
        events.extend(b.text_delta("after"))
        events.extend(b.finish())
        done_messages = [
            event.item
            for event in events
            if isinstance(event, ResponseOutputItemDoneEvent)
            and isinstance(event.item, OutputMessage)
        ]

        assert [msg.content[0].text for msg in done_messages] == ["before", "after"]

    def test_out_of_order_tool_chunks_keep_call_ids_in_provider_order(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=1, call_id="call_b", name="b", arguments="{}"))
        list(b.tool_call_delta(index=0, call_id="call_a", name="a", arguments="{}"))

        assert b.call_ids() == ["call_a", "call_b"]

    def test_server_tool_output_converts_tool_to_server_function_call(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments='{"x": 1}'))

        events = list(b.server_tool_output("call_1", "result"))

        assert isinstance(events[0], ResponseFunctionCallArgsDoneEvent)
        assert isinstance(events[1], ResponseOutputItemDoneEvent)
        assert isinstance(events[1].item, ServerFunctionCall)
        assert events[1].item.output == "result"

        completed = list(b.finish())[-1]
        assert isinstance(completed, ResponseCompletedEvent)
        assert len(completed.response.output) == 1
        assert isinstance(completed.response.output[0], ServerFunctionCall)

    def test_server_tool_output_before_tool_state_is_applied_when_tool_arrives(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        assert list(b.server_tool_output("call_1", "early")) == []

        events = list(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments="{}"))

        assert isinstance(events[-1], ResponseOutputItemDoneEvent)
        assert isinstance(events[-1].item, ServerFunctionCall)
        assert events[-1].item.output == "early"

    def test_duplicate_server_tool_output_raises(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_1", name="t", arguments="{}"))
        list(b.server_tool_output("call_1", "result"))

        with pytest.raises(RuntimeError, match="already finalized"):
            list(b.server_tool_output("call_1", "again"))

    def test_finish_twice_raises(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.finish())

        with pytest.raises(RuntimeError, match="terminal"):
            list(b.finish())


# ---------------------------------------------------------------------------
# call_ids
# ---------------------------------------------------------------------------


class TestCallIds:
    def test_returns_ids_in_order(self):
        b = ResponseBuilder(_make_response())
        list(b.start())
        list(b.tool_call_delta(index=0, call_id="call_a", name="a", arguments="{}"))
        list(b.tool_call_delta(index=1, call_id="call_b", name="b", arguments="{}"))
        assert b.call_ids() == ["call_a", "call_b"]

    def test_empty_when_no_tools(self):
        b = ResponseBuilder(_make_response())
        assert b.call_ids() == []
