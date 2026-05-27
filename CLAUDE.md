# Subspace

Streaming-first API middleware/proxy for LLM models and agents. Exposes pluggable API interfaces (OpenResponses, AG-UI) that route requests through a configurable middleware chain before hitting a backend (via litellm or custom).

## Architecture

```
Interface (OpenResponses, AG-UI, ...) → Middleware Chain (onion model) → Backend (litellm, echo, custom)
```

- **Interfaces** own routes and translate between wire format and internal `StreamEvent` types
- **Middlewares** wrap `call_next` — can modify the request inbound and transform the stream outbound
- **Backends** terminate the chain and produce `AsyncIterator[StreamEvent]`
- **`Subspace`** is the central registry: models/backends are shared, interfaces bring their own middleware stacks

## Project structure

- `src/subspace/core.py` — `Subspace` class (model registry, chain builder)
- `src/subspace/middleware/` — core framework (base protocol, callback, chain, context, stream_aggregator)
- `src/subspace/middleware/mcp/` — MCP server bridge (connects external MCP servers, exposes their tools)
- `src/subspace/middleware/conversation_history/` — core middleware with its own `storage/` adapters
- `src/subspace/contrib/middleware/` — shipped but non-core middlewares (delegate, instruction_injector, logging)
- `src/subspace/backends/` — chain terminators (litellm, echo)
- `src/subspace/interfaces/` — API surfaces (openresponses)
- `src/subspace/models/` — Pydantic v2 schemas (events, items, request, response, tools)

Middlewares that are generic framework concerns live in `middleware/`. Implementation-specific middlewares live in `contrib/middleware/`. Each middleware is a package when it has sub-components (storage adapters, helpers), otherwise a single file.

## Commands

```sh
uv run ruff check src/         # lint
uv run ruff format src/        # format
uv run pytest                  # tests
uv run python -m subspace      # run dev server
```

## Code style

- Python 3.13+, Pydantic v2, FastAPI, async throughout
- Ruff for linting and formatting — config in `pyproject.toml` (`target-version = "py313"`, line-length 100)
- Lint rules: `E`, `F`, `I`, `UP`, `B`, `SIM`, `TCH` — modern Python style, sorted imports, no unnecessary complexity
- No `from __future__ import annotations` — this is Python 3.13, not needed and it breaks runtime type access. Treat its presence as a code smell.
- No `getattr(obj, "field", ...)` for accessing known model fields — use direct attribute access. `getattr` on typed models is a code smell; if you need to check what type something is, use `isinstance`.
- No YAML/TOML config files for runtime — all configuration via Pydantic Settings + env vars
- Prefer `isinstance` checks over duck-typing with `hasattr`
- Use `X | Y` for type unions, not `Union[X, Y]`
- Middleware and backend `handle`/`__call__` methods are async generators (they `yield`). Protocol signatures must NOT use `async def` — use `def ... -> AsyncIterator[StreamEvent]` so the type checker sees the correct return type (an `AsyncIterator`, not a `Coroutine` wrapping one).
- Middleware signature: `def __call__(self, ctx: RequestContext, call_next: NextHandler) -> AsyncIterator[StreamEvent]`
- Backend signature: `def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]`
- `NextHandler` is `Callable[[RequestContext], AsyncIterator[StreamEvent]]` (not `Awaitable`)
