# Subspace

Subspace is streaming-first infrastructure for treating agents like models. It
does not try to replace LiteLLM, LangChain, Pydantic AI, CrewAI, LangGraph, or
hosted/sandboxed agent runtimes. It wraps them behind reusable interfaces,
middleware, and backend contracts.

## Architecture

```text
Interface / Router -> Middleware Chain -> Backend
```

- **Interfaces / routers** own wire formats and FastAPI routes. They translate
  between external APIs such as OpenResponses, Chat Completions, Anthropic
  Messages, ACP-style APIs, or AG-UI and the internal stream model.
- **Middlewares** wrap `call_next` in onion order. They can modify
  `RequestContext.request` before the backend runs, transform or observe the
  outbound stream, execute server-side tools, add history, or emit custom events.
- **Backends** terminate the chain. A backend can call LiteLLM, run a LangChain
  or LangGraph runnable, orchestrate multiple agents, or proxy to a provisioned
  runtime.
- **`Subspace`** is the central agent registry. Agents have a backend,
  middleware stack, metadata, runtime type, and effective capabilities.

## Current Project Structure

- `src/subspace/core.py` - `Subspace`, `Agent`, capability resolution, lifecycle
  entry for registered backends and middlewares.
- `src/subspace/backends/` - core backend contracts and built-in backends:
  `LitellmBackend`, `MultiAgentBackend`, and `ResponseBuilder`.
- `src/subspace/contrib/backends/` - optional backend integrations such as
  `LangchainBackend`.
- `src/subspace/fastapi/` - FastAPI app/mount helpers and routers.
- `src/subspace/fastapi/routers/` - OpenResponses, Chat Completions, Anthropic
  Messages, and shared router helpers.
- `src/subspace/middleware/` - framework-level middleware primitives and core
  middlewares: chain, context, function calls, MCP, stream helpers,
  conversation history, conditional middleware, stream tap.
- `src/subspace/middleware/utils/` - stream/event utilities such as
  `StreamTracker`.
- `src/subspace/contrib/middleware/` - useful but non-core middleware such as
  logging, instruction injection, Langfuse prompts, and retraction.
- `src/subspace/models/` - Pydantic v2 schemas for agents, capabilities,
  content, events, items, requests, responses, and tools.
- `src/subspace/streaming/` - stream serialization helpers such as SSE.
- `examples/` - runnable examples.
- `tests/` - unit tests for backends, middleware, routers, streaming, and core
  registry behavior.

Generic framework concerns belong in `middleware/`. Provider/framework-specific
integrations belong in `contrib/` unless they define a core Subspace contract.

## Commands

```sh
uv sync
uv run ruff check src tests examples
uv run ruff format src tests examples
uv run pytest
```

To run examples, prefer the example modules:

```sh
uv run python -m examples.simple
uv run python -m examples.delegate
uv run python -m examples.mcp
```

Do not advertise `uv run python -m subspace` unless `src/subspace/__main__.py`
exists again.

## Core Contracts

- `StreamEvent` is intentionally open. Custom middleware and routers may define
  additional event types.
- Use `BuiltInStreamEvent` when code only understands Subspace's built-in event
  set.
- Use `TerminalStreamEvent` or explicit `isinstance` checks for
  `ResponseCompletedEvent`, `ResponseFailedEvent`, and
  `ResponseIncompleteEvent`.
- Backend protocol:

  ```python
  def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]: ...
  ```

- Middleware signature:

  ```python
  def __call__(
      self,
      ctx: RequestContext,
      call_next: NextHandler,
  ) -> AsyncIterator[StreamEvent]: ...
  ```

- `NextHandler` is `Callable[[RequestContext], AsyncIterator[StreamEvent]]`.
  It is not awaitable.
- Middleware and backend streaming methods are async generators. Protocol
  signatures should use plain `def ... -> AsyncIterator[StreamEvent]`, not
  `async def`, so type checkers see an iterator rather than a coroutine.
- Backends may expose `capabilities: AgentCapabilities`.
- Middleware can implement `transform_capabilities()` to describe how it changes
  the effective chain.
- Capability fields are typed attributes. Avoid string-to-attribute mapping for
  capability checks.

## Lifecycle

- `Subspace.__aenter__()` enters unique registered backends that are async
  context managers, then unique registered agent middlewares.
- `SubspaceApp` also enters mount-level/interface middlewares and custom mount
  lifespan contexts.
- Middleware `prepare(ctx)` runs once per request before that middleware handles
  the stream.
- Middleware `finalize(ctx)` runs after the stream ends and is called in reverse
  middleware order for prepared middlewares.
- If a middleware conditionally runs another middleware, memoize the condition
  during `prepare()` and use the same decision during `__call__()` and
  `finalize()`.

## Stream Handling Patterns

- Use `ResponseBuilder` inside backends that translate provider-native streams
  into Subspace events. It owns lifecycle events, output indexes, item state,
  tool-call accumulation, and terminal completed/failed events.
- Use `StreamTracker` when middleware needs to track completed output items or
  close open items after cancellation/retraction.
- Routers should handle all terminal response events, including
  `ResponseFailedEvent`.
- Do not drop unknown custom stream events unless the router explicitly cannot
  represent them. Prefer pass-through for Subspace-native streams.
- Non-streaming router paths should collect the terminal response with shared
  helpers instead of open-coding event loops.

## Middleware Patterns

- Use `FunctionCallMiddleware` for server-side tool execution. Returning `None`
  from `on_function_call()` means the function call passes through to the
  client.
- `@server_tool` registers server-side tools and injects them into the request.
- Keep server-side tool calls visible as stream events. Use `ServerFunctionCall`
  when a middleware executed the tool.
- `FunctionCallMiddleware.max_tool_roundtrips` prevents unbounded server-tool
  loops.
- `StreamTapMiddleware` is for concurrent observation that should not block the
  stream. Consumer failures should be logged, not silently swallowed.
- `RetractionMiddleware` emits explicit retraction events when streamed output
  should be removed by the client.
- `ConversationHistoryMiddleware` stores copies of request/output items, not
  live object references.

## Router Patterns

- Routers live under `src/subspace/fastapi/routers/`.
- Keep route registration, inbound translation, outbound streaming translation,
  non-streaming translation, and error formatting separated when a router grows.
- Use shared error/terminal helpers from `routers/_shared.py` where possible.
- OpenResponses can expose Subspace events directly.
- Chat Completions and Anthropic Messages are compatibility routers; they are
  necessarily lossy. Be explicit about what cannot be represented.
- Router compatibility should eventually be validated against effective agent
  capabilities, not inferred from strings or duck-typing.

## Backend Patterns

- Built-in model/provider backends belong in `src/subspace/backends/`.
- Framework-specific agent integrations belong in `src/subspace/contrib/backends/`.
- `LitellmBackend` is the simple model backend.
- `LangchainBackend` wraps a LangChain/LangGraph runnable or a factory that
  receives `(ctx, interrupt_tools)`.
- `MultiAgentBackend` is opinionated whole-conversation handoff, not A2A. It
  injects a `delegate_to` tool and hides that tool call from the client stream.
- Provisioned/sandboxed agents should be modeled as backends with lifecycle,
  runtime capabilities, permissions, and control-plane events instead of being
  forced into Chat Completions or Anthropic message shapes.

## Code Style

- Python 3.13+, Pydantic v2, FastAPI, async throughout.
- Ruff config lives in `pyproject.toml` (`target-version = "py313"`,
  line-length 100).
- Lint rules: `E`, `F`, `I`, `UP`, `B`, `SIM`, `TCH`.
- No `from __future__ import annotations`. Runtime type access matters here.
- Use `X | Y` unions, not `Union[X, Y]`.
- Use `Field(default_factory=list)` for mutable Pydantic defaults.
- Prefer `isinstance` checks over duck typing with `hasattr`.
- Do not use `getattr(obj, "field", ...)` for known typed model fields. Use
  direct attribute access. If the type is uncertain, narrow it with
  `isinstance`.
- Avoid stringly-typed capability checks. Capabilities are direct typed model
  fields.
- No YAML/TOML runtime config. Prefer Python objects, Pydantic models, and env
  vars where configuration is needed.
- Public methods should have at least a short one-line docstring.
- Do not leave raw `print()` debugging in library code. Use module loggers.

## Testing Expectations

- Run `uv run ruff check src tests examples` and `uv run pytest` before handing
  off a change.
- Backend changes need stream lifecycle tests: created, in-progress, deltas,
  item completion, terminal response, usage, and failure.
- Router changes need both streaming and non-streaming tests, plus wire-format
  error tests.
- Middleware changes need ordering, pass-through/suppression, terminal event,
  error, and cancellation/retraction tests where relevant.
- Capability/lifecycle changes need tests in `tests/test_core.py`.
- If a behavior is intentionally lossy or opinionated, test that explicitly.

## Documentation Expectations

- README examples should be small and runnable.
- Keep claims conservative. If a feature only works for one provider, backend,
  router, or runtime, say so.
- Use "agent" language for registry concepts. "Model" is the client-facing
  compatibility shape, not the internal registry abstraction.
- Mention opinionated features as opinionated, especially multi-agent handoff
  and retraction events.
