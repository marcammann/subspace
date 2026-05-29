"""Delegation example: a multi-agent backend delegates to specialists.

A multi-agent backend fronts two specialists: a weather agent and a time agent.
Only the public "assistant" agent is exposed through the API — the specialists
are internal.

Run:
    uv run python -m examples.delegate

Then:
    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "assistant", "input": "What is the weather in Berlin?", "stream": true}'

    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "assistant", "input": "What time is it?", "stream": true}'
"""

import logging

import uvicorn
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from subspace import (
    OpenResponsesRouter,
    RequestContext,
    Subspace,
    SubspaceApp,
    SubspaceMount,
)
from subspace.backends.multi_agent import MultiAgentBackend
from subspace.contrib.backends.langchain import LangchainBackend
from subspace.contrib.middleware.logging import LoggingMiddleware


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"


@tool
def get_time() -> str:
    """Get the current UTC time."""
    return "12:00"


def make_weather(ctx: RequestContext, interrupt_tools: list):
    in_memory_saver = InMemorySaver()

    return create_react_agent(
        "anthropic:claude-sonnet-4-6",
        prompt="You are a weather assistant. Use the get_weather tool to answer weather questions. "
        "If asked about something other than weather, delegate to a more appropriate agent.",
        tools=[get_weather] + interrupt_tools,
        checkpointer=in_memory_saver,
    )


def make_time(ctx: RequestContext, interrupt_tools: list):
    in_memory_saver = InMemorySaver()

    return create_react_agent(
        "anthropic:claude-sonnet-4-6",
        prompt="You are a time assistant. Use the get_time tool to answer time questions. "
        "If asked about something other than time, delegate to a more appropriate agent.",
        tools=[get_time] + interrupt_tools,
        checkpointer=in_memory_saver,
    )


def main() -> None:
    # Internal agents — not exposed through the API
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    internal = Subspace()
    weather = internal.agent(
        "weather",
        backend=LangchainBackend(make_weather),
        middlewares=[LoggingMiddleware()],
        description="Answers weather questions for any city.",
    )
    time_agent = internal.agent(
        "time",
        backend=LangchainBackend(make_time),
        middlewares=[LoggingMiddleware()],
        description="Returns the current UTC time.",
    )

    # Only the multi-agent entry point is public
    mount = SubspaceMount(
        interfaces=[OpenResponsesRouter(prefix="/v1")],
    )
    mount.agent(
        "assistant",
        backend=MultiAgentBackend(agents=[weather, time_agent]),
    )

    app = SubspaceApp(mount, title="Subspace Delegate Example")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
