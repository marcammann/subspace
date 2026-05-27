import inspect
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, get_type_hints

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.common import Status, Usage
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import (
    FunctionCall,
    FunctionCallOutput,
    OutputMessage,
    ServerFunctionCall,
)
from subspace.models.tools import FunctionTool

_SERVER_TOOL_ATTR = "_server_tool"


def server_tool(
    name: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable:
    """Decorator that registers a method as a server-side tool.

    The tool schema is inferred from the function signature using Annotated types:

        class MyMiddleware(FunctionCallMiddleware):
            @server_tool(name="get_time", description="Get current UTC time")
            async def get_time(
                self,
                ctx: RequestContext,
                timezone: Annotated[str, Field(description="IANA timezone")] = "UTC",
            ) -> str:
                return datetime.now(ZoneInfo(timezone)).isoformat()

    Parameters `self` and `ctx` are skipped. Remaining parameters become the tool's
    JSON schema. Use Annotated[type, Field(...)] to add descriptions and constraints.

    Pass `parameters` explicitly to override schema inference.
    """

    def decorator(fn: Callable) -> Callable:
        schema = parameters if parameters is not None else _schema_from_signature(fn)
        setattr(
            fn,
            _SERVER_TOOL_ATTR,
            FunctionTool(
                name=name,
                description=description,
                parameters=schema,
            ),
        )
        return fn

    return decorator


class FunctionCallMiddleware(Middleware):
    """Middleware that handles function calls server-side.

    Server-tool dispatch, round-trip loops, and the @server_tool decorator
    for declarative server-side functions.

    Decorated @server_tool methods are automatically:
    - Injected into the request as available tools
    - Handled server-side when the model calls them
    - Emitted as ServerFunctionCall items (not FunctionCall) so clients won't execute them
    - Fed back to the model for another round-trip (agentic loop)

    For functions that need more control, override on_function_call directly.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        functions: dict[str, tuple[FunctionTool, Callable]] = {}
        for attr in cls.__dict__.values():
            if callable(attr) and hasattr(attr, _SERVER_TOOL_ATTR):
                tool_def: FunctionTool = getattr(attr, _SERVER_TOOL_ATTR)
                functions[tool_def.name] = (tool_def, attr)
        parent_functions = getattr(cls, "_registered_functions", {})
        cls._registered_functions = {**parent_functions, **functions}

    _registered_functions: dict[str, tuple[FunctionTool, Callable]] = {}

    async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
        """Called when an owned function call completes (after @server_tool dispatch).

        Return a string to execute the function server-side and trigger a round-trip.
        Return None to pass the function call through to the client.
        """
        return None

    async def prepare(self, ctx: RequestContext) -> None:
        if self._registered_functions:
            functions_to_inject = [tool for tool, _ in self._registered_functions.values()]
            existing = ctx.request.tools or []
            ctx.request = ctx.request.model_copy(update={"tools": existing + functions_to_inject})

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        ctx.request = ctx.request.model_copy(
            update={"input": self._unbundle_server_calls(ctx.request.input)}
        )

        all_output: list[OutputMessage | FunctionCall | ServerFunctionCall] = []
        total_usage = Usage()
        seq = 0
        round_num = 0

        while True:
            results: dict[str, str] = {}
            round_items: list[OutputMessage | FunctionCall | ServerFunctionCall] = []
            server_calls: dict[str, ServerFunctionCall] = {}

            fc_buffers: dict[str, list[StreamEvent]] = {}
            passthrough_ids: set[str] = set()

            async for event in call_next(ctx):
                seq = max(seq, event.sequence_number) + 1

                if round_num > 0 and isinstance(
                    event, (ResponseCreatedEvent, ResponseInProgressEvent)
                ):
                    continue

                if round_num > 0:
                    event = _offset_output_index(event, len(all_output))

                if isinstance(event, ResponseCompletedEvent):
                    if event.response.usage:
                        u = event.response.usage
                        total_usage = Usage(
                            input_tokens=total_usage.input_tokens + u.input_tokens,
                            output_tokens=total_usage.output_tokens + u.output_tokens,
                            total_tokens=total_usage.total_tokens + u.total_tokens,
                        )
                    continue

                # FunctionCall started — buffer if we own it, pass through otherwise
                if isinstance(event, ResponseOutputItemAddedEvent) and isinstance(
                    event.item, FunctionCall
                ):
                    if self.owns_function(event.item.name):
                        fc_buffers[event.item.id] = [event]
                    else:
                        passthrough_ids.add(event.item.id)
                        yield event
                    continue

                # Buffered server-side function call events
                if _is_function_call_event(event, fc_buffers):
                    item_id = _get_item_id(event)
                    if item_id and item_id in fc_buffers:
                        fc_buffers[item_id].append(event)

                    if isinstance(event, ResponseOutputItemDoneEvent):
                        item = event.item
                        if isinstance(item, FunctionCall):
                            result = await self._dispatch_function_call(ctx, item)
                            if result is not None:
                                results[item.call_id] = result
                                server_call = ServerFunctionCall(
                                    id=item.id,
                                    name=item.name,
                                    call_id=item.call_id,
                                    arguments=item.arguments,
                                    output=result,
                                )
                                server_calls[item.call_id] = server_call
                                yield ResponseOutputItemDoneEvent(
                                    sequence_number=seq,
                                    output_index=event.output_index,
                                    item=server_call,
                                )
                                round_items.append(item)

                        if item_id:
                            fc_buffers.pop(item_id, None)
                    continue

                # Passthrough client-side function call events
                if _is_function_call_event(event, passthrough_ids):
                    if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                        event.item, FunctionCall
                    ):
                        passthrough_ids.discard(event.item.id)
                        round_items.append(event.item)
                    yield event
                    continue

                if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                    event.item, OutputMessage
                ):
                    round_items.append(event.item)

                yield event

            for item in round_items:
                if isinstance(item, FunctionCall) and item.call_id in server_calls:
                    all_output.append(server_calls[item.call_id])
                else:
                    all_output.append(item)

            if not results:
                break

            current_input = ctx.request.input
            outputs = [
                FunctionCallOutput(call_id=item.call_id, output=results[item.call_id])
                for item in round_items
                if isinstance(item, FunctionCall) and item.call_id in results
            ]
            ctx.request = ctx.request.model_copy(
                update={"input": current_input + round_items + outputs}
            )
            round_num += 1

        yield ResponseCompletedEvent(
            sequence_number=seq,
            response=ctx.response.model_copy(
                update={
                    "status": Status.COMPLETED,
                    "output": all_output,
                    "usage": total_usage,
                }
            ),
        )

    def owns_function(self, name: str) -> bool:
        return name in self._registered_functions

    async def _dispatch_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
        # This isn't checking if we own the function, this is checking if the function
        # was registered using @server_tool — if so it's executed. Otherwise we call
        # `on_function_call`.
        if call.name in self._registered_functions:
            _, handler = self._registered_functions[call.name]
            kwargs = json.loads(call.arguments) if call.arguments else {}
            return await handler(self, ctx, **kwargs)
        return await self.on_function_call(ctx, call)

    def _unbundle_server_calls(self, input: list) -> list:
        """
        Server Function Call contains the function call and response in a single item as it's not a
        regular function call from a model, it's a function call the model executed server side.

        This unbundles a ServerFunctionCall and turns it into a set of FunctionCall & FunctionCallOutput,
        which is then sent "inwards".
        """
        result: list = []
        for item in input:
            if isinstance(item, ServerFunctionCall) and self.owns_function(item.name):
                result.append(
                    FunctionCall(
                        id=item.id,
                        name=item.name,
                        call_id=item.call_id,
                        arguments=item.arguments,
                        status=Status.COMPLETED,
                    )
                )
                result.append(FunctionCallOutput(call_id=item.call_id, output=item.output))
            else:
                result.append(item)
        return result


def _schema_from_signature(fn: Callable) -> dict[str, Any]:
    """Build a JSON schema from a function's type-annotated parameters."""
    from pydantic import create_model

    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)

    fields: dict[str, Any] = {}
    skip = {"self", "ctx", "return"}

    for param_name, param in sig.parameters.items():
        if param_name in skip:
            continue
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    if not fields:
        return {"type": "object", "properties": {}}

    model = create_model(f"{fn.__name__}_args", **fields)
    return model.model_json_schema()


def _is_function_call_event(event: StreamEvent, fc_ids: dict[str, Any] | set[str]) -> bool:
    if isinstance(event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent)):
        return isinstance(event.item, FunctionCall) and event.item.id in fc_ids
    if isinstance(event, (ResponseFunctionCallArgsDeltaEvent, ResponseFunctionCallArgsDoneEvent)):
        return event.item_id in fc_ids
    return False


def _get_item_id(event: StreamEvent) -> str | None:
    if isinstance(
        event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent)
    ) and isinstance(event.item, FunctionCall):
        return event.item.id
    if isinstance(event, (ResponseFunctionCallArgsDeltaEvent, ResponseFunctionCallArgsDoneEvent)):
        return event.item_id
    return None


def _offset_output_index(event: StreamEvent, offset: int) -> StreamEvent:
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
