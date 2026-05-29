import json
import uuid

import pytest
from conftest import StaticBackend, make_response, text_backend_events

from subspace.backends.multi_agent import MultiAgentBackend
from subspace.core import Subspace
from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.models.common import Status, Usage
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, FunctionCallOutput, OutputMessage
from subspace.models.request import CreateResponseRequest


def _delegate_call_events(target: str) -> list[StreamEvent]:
    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
    call_id = "call_delegate"
    arguments = json.dumps({"agent": target})
    seq = 0
    response = make_response()
    events: list[StreamEvent] = []

    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1

    fc = FunctionCall(
        id=fc_id, name="delegate_to", call_id=call_id,
        arguments="", status=Status.IN_PROGRESS,
    )
    events.append(
        ResponseOutputItemAddedEvent(sequence_number=seq, output_index=0, item=fc)
    )
    seq += 1
    events.append(
        ResponseOutputItemDoneEvent(
            sequence_number=seq, output_index=0,
            item=FunctionCall(
                id=fc_id, name="delegate_to", call_id=call_id,
                arguments=arguments, status=Status.COMPLETED,
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


class RecordingBackend:
    def __init__(self, events_per_call: list[list[StreamEvent]]) -> None:
        self._events_per_call = events_per_call
        self.calls: list[CreateResponseRequest] = []

    async def handle(self, ctx: RequestContext):
        self.calls.append(ctx.request)
        call_idx = len(self.calls) - 1
        if call_idx < len(self._events_per_call):
            for event in self._events_per_call[call_idx]:
                yield event


def _make_ctx(model: str = "assistant") -> RequestContext:
    request = CreateResponseRequest(model=model, input="hello")
    response = make_response(model=model)
    return RequestContext(request=request, response_id="resp_test", response=response)


class TestMultiAgentBackend:
    def test_requires_at_least_one_agent(self):
        with pytest.raises(ValueError, match="at least one agent"):
            MultiAgentBackend(agents=[])

    @pytest.mark.anyio
    async def test_defaults_to_first_agent(self):
        s = Subspace()
        weather = s.agent(
            "weather", backend=StaticBackend(text_backend_events("Sunny")),
            description="Weather",
        )
        time_agent = s.agent(
            "time", backend=StaticBackend(text_backend_events("12:00")),
            description="Time",
        )

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1

        output_msgs = [
            item for item in completed[0].response.output
            if isinstance(item, OutputMessage)
        ]
        assert len(output_msgs) > 0

    @pytest.mark.anyio
    async def test_delegation_sequence_numbers_are_monotonic(self):
        weather_backend = RecordingBackend([
            _delegate_call_events("time"),
            text_backend_events("OK"),
        ])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")
        time_agent = s.agent(
            "time", backend=StaticBackend(text_backend_events("It is noon")),
            description="Time",
        )

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        sequence_numbers = [event.sequence_number for event in events]

        assert sequence_numbers == list(range(len(sequence_numbers)))

    @pytest.mark.anyio
    async def test_multiple_delegations(self):
        weather_backend = RecordingBackend([
            _delegate_call_events("time"),
            text_backend_events("weather closed"),
        ])
        time_backend = RecordingBackend([
            _delegate_call_events("trivia"),
            text_backend_events("time closed"),
        ])
        trivia_backend = RecordingBackend([text_backend_events("final answer")])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")
        time_agent = s.agent("time", backend=time_backend, description="Time")
        trivia = s.agent("trivia", backend=trivia_backend, description="Trivia")

        backend = MultiAgentBackend(agents=[weather, time_agent, trivia])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]

        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED
        assert len(weather_backend.calls) == 2
        assert len(time_backend.calls) == 2
        assert len(trivia_backend.calls) == 1

        output_text = [
            part.text
            for item in completed[0].response.output
            if isinstance(item, OutputMessage)
            for part in item.content
        ]
        assert output_text == ["final answer"]

    @pytest.mark.anyio
    async def test_max_delegations_exceeded(self):
        weather_backend = RecordingBackend([
            _delegate_call_events("time"),
            text_backend_events("weather closed"),
        ])
        time_backend = RecordingBackend([_delegate_call_events("weather")])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")
        time_agent = s.agent("time", backend=time_backend, description="Time")

        backend = MultiAgentBackend(agents=[weather, time_agent], max_delegations=1)
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseFailedEvent)]

        assert len(completed) == 1
        assert completed[0].response.status == Status.FAILED
        assert completed[0].response.error.code == "max_delegations_exceeded"

    @pytest.mark.anyio
    async def test_delegation_streams_target_response(self):
        weather_backend = RecordingBackend([
            _delegate_call_events("time"),
            text_backend_events("OK"),
        ])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")
        time_agent = s.agent(
            "time", backend=StaticBackend(text_backend_events("It is noon")),
            description="Time",
        )

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED

        output_msgs = [
            item for item in completed[0].response.output
            if isinstance(item, OutputMessage)
        ]
        assert len(output_msgs) > 0

    @pytest.mark.anyio
    async def test_close_out_before_delegation(self):
        weather_backend = RecordingBackend([
            _delegate_call_events("time"),
            text_backend_events("OK"),
        ])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")
        time_agent = s.agent(
            "time", backend=StaticBackend(text_backend_events("Noon")),
            description="Time",
        )

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        async for _ in chain.execute(ctx):
            pass

        # Weather backend called twice: initial + close-out (before delegation)
        assert len(weather_backend.calls) == 2

        close_request = weather_backend.calls[1]
        fc_outputs = [
            item for item in close_request.input
            if isinstance(item, FunctionCallOutput)
        ]
        assert len(fc_outputs) == 1
        assert "Delegated to time" in fc_outputs[0].output
        assert close_request.tools is None
        assert close_request.max_output_tokens == 16

    @pytest.mark.anyio
    async def test_unknown_target_error(self):
        weather_backend = RecordingBackend([_delegate_call_events("nonexistent")])

        s = Subspace()
        weather = s.agent("weather", backend=weather_backend, description="Weather")

        backend = MultiAgentBackend(agents=[weather])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseFailedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.FAILED

        # No close-out
        assert len(weather_backend.calls) == 1

    @pytest.mark.anyio
    async def test_no_delegation_passes_through(self):
        s = Subspace()
        weather = s.agent(
            "weather", backend=StaticBackend(text_backend_events("Sunny")),
            description="Weather",
        )
        time_agent = s.agent(
            "time", backend=StaticBackend(text_backend_events("Noon")),
            description="Time",
        )

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        events = [e async for e in chain.execute(ctx)]
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED

    @pytest.mark.anyio
    async def test_delegate_tool_injected(self):
        recording = RecordingBackend([text_backend_events("Hi")])

        s = Subspace()
        weather = s.agent("weather", backend=recording, description="Weather info")
        time_agent = s.agent("time", backend=StaticBackend(text_backend_events()), description="Time")

        backend = MultiAgentBackend(agents=[weather, time_agent])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        async for _ in chain.execute(ctx):
            pass

        request = recording.calls[0]
        tool_names = [t.name for t in (request.tools or [])]
        assert "delegate_to" in tool_names

        delegate_tool = next(t for t in request.tools if t.name == "delegate_to")
        agent_enum = delegate_tool.parameters["properties"]["agent"]["enum"]
        assert "time" in agent_enum
        assert "weather" not in agent_enum

    @pytest.mark.anyio
    async def test_delegate_can_hand_back(self):
        """The delegate agent can hand off to any other agent including the originator."""
        time_recording = RecordingBackend([text_backend_events("It is noon")])

        s = Subspace()
        weather = s.agent(
            "weather",
            backend=RecordingBackend([
                _delegate_call_events("time"),
                text_backend_events("OK"),
            ]),
            description="Weather",
        )
        time_agent = s.agent("time", backend=time_recording, description="Time")
        trivia = s.agent("trivia", backend=StaticBackend(text_backend_events()), description="Trivia")

        backend = MultiAgentBackend(agents=[weather, time_agent, trivia])
        chain = MiddlewareChain(middlewares=[], backend=backend)
        ctx = _make_ctx()

        async for _ in chain.execute(ctx):
            pass

        # The delegate (time) should have a delegate_to tool that excludes
        # only itself — it can hand back to weather or forward to trivia.
        delegate_request = time_recording.calls[0]
        delegate_tools = [t for t in (delegate_request.tools or []) if t.name == "delegate_to"]
        assert len(delegate_tools) == 1
        agent_enum = delegate_tools[0].parameters["properties"]["agent"]["enum"]
        assert "weather" in agent_enum
        assert "trivia" in agent_enum
        assert "time" not in agent_enum
