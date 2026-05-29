import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.middleware.utils.events import offset_output_index
from subspace.models.agent import AgentCapabilities
from subspace.models.common import ResponseError, Role, Status, Usage
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, FunctionCallOutput, InputMessage, OutputMessage
from subspace.models.tools import FunctionTool

if TYPE_CHECKING:
    from subspace.core import Agent

TOOL_NAME = "delegate_to"


class MultiAgentBackend:
    """Backend that lets a group of agents hand off to each other."""

    def __init__(self, agents: list["Agent"], *, max_delegations: int = 8) -> None:
        if not agents:
            msg = "MultiAgentBackend requires at least one agent"
            raise ValueError(msg)
        if max_delegations < 0:
            msg = "max_delegations must be non-negative"
            raise ValueError(msg)
        self._agents: dict[str, Agent] = {agent.name: agent for agent in agents}
        self._default = agents[0]
        self._max_delegations = max_delegations

    @property
    def capabilities(self) -> AgentCapabilities:
        """Capabilities exposed by the multi-agent orchestration backend."""
        return AgentCapabilities(
            streaming=True,
            text_input=True,
            function_tools=True,
            delegation=True,
            multiple_delegations=self._max_delegations > 1,
        )

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        existing_tools = list(ctx.request.tools or [])
        state = _RunState()
        active = self._default
        active_ctx = ctx
        delegation_count = 0
        skip_lifecycle = False

        while True:
            self._inject_delegate_tool(
                active_ctx,
                exclude={active.name},
                existing_tools=existing_tools,
            )
            active_chain = active.build_chain()
            output_start = len(state.output)
            interceptor = _DelegateInterceptor()

            async for event in _stream_agent(
                chain=active_chain,
                ctx=active_ctx,
                state=state,
                delegate_interceptor=interceptor,
                output_offset=output_start,
                skip_lifecycle=skip_lifecycle,
            ):
                yield event

            if interceptor.delegate_call is None:
                break

            if delegation_count >= self._max_delegations:
                yield _failed_event(
                    ctx=ctx,
                    state=state,
                    message="Maximum delegation count exceeded",
                    code="max_delegations_exceeded",
                )
                return

            handoff = await self._prepare_handoff(
                ctx=active_ctx,
                active=active,
                active_chain=active_chain,
                delegate_call=interceptor.delegate_call,
                existing_tools=existing_tools,
                state=state,
                output_start=output_start,
            )
            if handoff is None:
                yield _failed_event(
                    ctx=ctx,
                    state=state,
                    message=f"Cannot delegate: agent '{_parse_target(interceptor.delegate_call)}' not found",
                    code="agent_not_found",
                )
                return

            active, active_ctx = handoff
            delegation_count += 1
            skip_lifecycle = True

        yield _completed_event(ctx, state)

    async def _prepare_handoff(
        self,
        *,
        ctx: RequestContext,
        active: "Agent",
        active_chain: MiddlewareChain,
        delegate_call: FunctionCall,
        existing_tools: list[FunctionTool],
        state: "_RunState",
        output_start: int,
    ) -> tuple["Agent", RequestContext] | None:
        target_name = _parse_target(delegate_call)
        agent = self._agents.get(target_name) if target_name else None

        if agent is None:
            return None

        current_output = state.output[output_start:]

        close_usage = await self._close_tool_call(
            chain=active_chain,
            ctx=ctx,
            delegate_call=delegate_call,
            prior_output=current_output,
            output=f"Delegated to {target_name}.",
        )
        state.usage = _add_usage(state.usage, close_usage)

        delegate_ctx = self._build_delegate_context(
            ctx=ctx,
            source=active,
            target=agent,
            existing_tools=existing_tools,
            prior_output=current_output,
        )
        return agent, delegate_ctx

    async def _close_tool_call(
        self,
        *,
        chain: MiddlewareChain,
        ctx: RequestContext,
        delegate_call: FunctionCall,
        prior_output: list[OutputMessage | FunctionCall],
        output: str,
    ) -> Usage:
        close_input: list[Any] = list(ctx.request.input) + list(prior_output)
        close_input.extend(
            [
                FunctionCall(
                    id=delegate_call.id,
                    name=delegate_call.name,
                    call_id=delegate_call.call_id,
                    arguments=delegate_call.arguments,
                    status=Status.COMPLETED,
                ),
                FunctionCallOutput(call_id=delegate_call.call_id, output=output),
            ]
        )

        close_request = ctx.request.model_copy(
            update={"input": close_input, "tools": None, "max_output_tokens": 16}
        )
        close_ctx = RequestContext(
            request=close_request,
            response_id=ctx.response_id,
            response=ctx.response,
            app=ctx.app,
            deps=ctx.deps,
            metadata=ctx.metadata,
        )
        # Share backend state so close-out can resume graph interrupts.
        close_ctx._state = ctx._state  # noqa: SLF001

        usage = Usage()
        async for event in chain.execute(close_ctx):
            if isinstance(event, ResponseCompletedEvent) and event.response.usage:
                usage = _add_usage(usage, event.response.usage)
        return usage

    def _build_delegate_context(
        self,
        *,
        ctx: RequestContext,
        source: "Agent",
        target: "Agent",
        existing_tools: list[FunctionTool],
        prior_output: list[OutputMessage | FunctionCall],
    ) -> RequestContext:
        delegate_input: list[Any] = list(ctx.request.input) + list(prior_output)
        if prior_output:
            delegate_input.append(
                InputMessage(role=Role.USER, content="[Conversation delegated to you]")
            )

        delegate_tools = [tool for tool in existing_tools if tool.name != TOOL_NAME]

        delegate_request = ctx.request.model_copy(
            update={
                "model": target.name,
                "input": delegate_input,
                "tools": delegate_tools,
            }
        )
        return RequestContext(
            request=delegate_request,
            response_id=ctx.response_id,
            response=ctx.response.model_copy(update={"model": target.name}),
            app=ctx.app,
            deps=ctx.deps,
            metadata={**ctx.metadata, "delegated_from": source.name},
        )

    def _inject_delegate_tool(
        self,
        ctx: RequestContext,
        *,
        exclude: set[str],
        existing_tools: list[FunctionTool],
    ) -> None:
        tool = self._build_tool(exclude=exclude)
        if tool:
            ctx.request = ctx.request.model_copy(update={"tools": existing_tools + [tool]})

    def _build_tool(self, exclude: set[str]) -> FunctionTool | None:
        agents = {name: agent for name, agent in self._agents.items() if name not in exclude}
        if not agents:
            return None

        agent_names = list(agents.keys())
        description = "\n\n".join(_agent_description(name, agent) for name, agent in agents.items())

        return FunctionTool(
            name=TOOL_NAME,
            description=(
                "Hand off the entire conversation to another agent that is better suited "
                "to respond. Only call to hand off the entire conversation, not subtasks. "
                "You MUST NOT mention that you're handing the conversation to another agent.\n\n"
                + description
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": agent_names,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this agent is better suited.",
                    },
                },
                "required": ["agent"],
                "additionalProperties": False,
            },
        )


@dataclass
class _RunState:
    output: list[OutputMessage | FunctionCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    sequence_number: int = 0

    def prepare(self, event: StreamEvent) -> StreamEvent:
        event = event.model_copy(update={"sequence_number": self.sequence_number})
        self.sequence_number += 1
        return event


@dataclass
class _DelegateInterceptor:
    item_ids: set[str] = field(default_factory=set)
    delegate_call: FunctionCall | None = None

    def suppresses(self, event: StreamEvent) -> bool:
        if (
            isinstance(event, ResponseOutputItemAddedEvent)
            and isinstance(event.item, FunctionCall)
            and event.item.name == TOOL_NAME
        ):
            self.item_ids.add(event.item.id)
            return True

        if not _is_delegate_event(event, self.item_ids):
            return False

        if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(event.item, FunctionCall):
            self.delegate_call = event.item
        return True


async def _stream_agent(
    *,
    chain: MiddlewareChain,
    ctx: RequestContext,
    state: _RunState,
    delegate_interceptor: _DelegateInterceptor | None = None,
    output_offset: int = 0,
    skip_lifecycle: bool = False,
) -> AsyncIterator[StreamEvent]:
    async for event in chain.execute(ctx):
        if skip_lifecycle and isinstance(event, (ResponseCreatedEvent, ResponseInProgressEvent)):
            continue

        if delegate_interceptor and delegate_interceptor.suppresses(event):
            continue

        if isinstance(event, ResponseCompletedEvent):
            if event.response.usage:
                state.usage = _add_usage(state.usage, event.response.usage)
            continue

        if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
            event.item, (OutputMessage, FunctionCall)
        ):
            state.output.append(event.item)

        event = offset_output_index(event, output_offset)
        yield state.prepare(event)


def _completed_event(ctx: RequestContext, state: _RunState) -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        sequence_number=state.sequence_number,
        response=ctx.response.model_copy(
            update={
                "status": Status.COMPLETED,
                "output": state.output,
                "usage": state.usage,
            }
        ),
    )


def _failed_event(
    ctx: RequestContext,
    state: _RunState,
    message: str,
    code: str,
) -> ResponseFailedEvent:
    return ResponseFailedEvent(
        sequence_number=state.sequence_number,
        response=ctx.response.model_copy(
            update={
                "status": Status.FAILED,
                "output": state.output,
                "usage": state.usage,
                "error": ResponseError(
                    message=message,
                    type="invalid_request",
                    code=code,
                ),
            }
        ),
    )


def _is_delegate_event(event: StreamEvent, delegate_ids: set[str]) -> bool:
    if (
        isinstance(event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent))
        and isinstance(event.item, (FunctionCall, OutputMessage))
        and event.item.id in delegate_ids
    ):
        return True
    return (
        isinstance(event, (ResponseFunctionCallArgsDeltaEvent, ResponseFunctionCallArgsDoneEvent))
        and event.item_id in delegate_ids
    )


def _parse_target(call: FunctionCall) -> str | None:
    try:
        args = json.loads(call.arguments)
        return args.get("agent")
    except (json.JSONDecodeError, AttributeError):
        return None


def _agent_description(name: str, agent: "Agent") -> str:
    card = agent.card
    lines = [f"# {name}"]
    if card.description:
        lines.append(f"description: {card.description}")
    if card.skills:
        lines.append("skills: " + ", ".join(skill.name for skill in card.skills))
    return "\n".join(lines)


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )
