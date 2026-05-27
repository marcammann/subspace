# Backend / Middleware Compatibility

Not all middlewares work with all backend types. The key distinction is **who owns the agentic loop** — the middleware chain or the backend.

## Backend Types

| Backend | Loop Owner | Description |
|---|---|---|
| **LitellmBackend** | Middleware | Calls the LLM once, emits `FunctionCall` events. Middleware handles tool execution and round-trips. |
| **EchoBackend** | Middleware | Echoes input. No tool support. |
| **LangchainBackend (chat model)** | Middleware | Wraps a `BaseChatModel`. Same as Litellm — single LLM call, middleware handles the loop. |
| **LangchainBackend (agent)** | Agent | Wraps a `create_agent()` graph. Agent runs its own tool loop internally. Tool calls appear as `ServerToolCall` in the output. |

## Middleware Compatibility

| Middleware | Chat Model Backends | Agent Backends | Notes |
|---|---|---|---|
| **LoggingMiddleware** | ✓ | ✓ | Passive — observes events, doesn't modify them. |
| **InstructionInjectorMiddleware** | ✓ | ✓ | Sets `ctx.request.instructions`. Agent factory must read it. |
| **ConversationHistoryMiddleware** | ✓ | ✓ | Loads/saves conversation history on the request. |
| **StreamAggregatorMiddleware** | ✓ | ✓ | Buffers the stream — works with any backend. |
| **DelegateMiddleware** | ✓ | ✗ | Injects a tool and handles it server-side. Agent won't emit `FunctionCall` for it. |
| **CallbackMiddleware / @server_tool** | ✓ | ✗ | Intercepts `FunctionCall` events. Agent executes tools internally, never emits them. |
| **McpMiddleware** | ✓ | ✗ | Same as @server_tool — injects tools and intercepts calls. |
| **LangfusePromptMiddleware** | ✓ | ✓ | Sets instructions/input. Agent factory must read `ctx.request.instructions`. |

## Why Agent Backends Don't Support Tool-Injecting Middlewares

Tool-injecting middlewares (`@server_tool`, `McpMiddleware`, `DelegateMiddleware`) work by:

1. Adding tool definitions to `ctx.request.tools`
2. Intercepting `FunctionCall` events from the backend
3. Executing the tool server-side and feeding the result back

Agent backends (e.g. `create_agent()`) own their own tool loop:

1. The model decides to call a tool
2. The agent's `ToolNode` executes it immediately
3. The result is fed back to the model internally
4. Only the final output is streamed out

Since the agent never emits `FunctionCall` events for its tools, middlewares can't intercept them. The agent's tools must be defined as LangChain tools and passed to `create_agent()` at construction time.

## Choosing the Right Pattern

**Use a chat model backend + middleware when you want:**
- Dynamic tool injection from middleware (MCP servers, server tools)
- Mix of server-side and client-side tools
- Tools that might be deferred to the client as `FunctionCall`
- Control over the agentic loop (max iterations, custom routing)

**Use an agent backend when you want:**
- LangChain's agent loop (automatic tool execution, state management)
- LangGraph features (checkpointing, human-in-the-loop, branching)
- Tools defined entirely in LangChain
- Middleware limited to instructions, logging, history, and other non-tool concerns
