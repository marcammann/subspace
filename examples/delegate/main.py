"""Delegation example: a triage agent delegates to specialists.

Three LangChain agents: a triage agent that delegates to either a
weather agent or a time agent depending on the question.

Run:
    uv run python -m examples.delegate

Then:
    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "triage", "input": "What time is it?", "stream": true}'

    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "triage", "input": "What is the weather in Berlin?", "stream": true}'
"""

from datetime import UTC, datetime

import uvicorn
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from subspace import (
    OpenResponsesInterface,
    RequestContext,
    SubspaceApp,
    SubspaceMount,
)
from subspace.contrib.backends.langchain import LangchainBackend
from subspace.contrib.middleware.delegate import DelegateMiddleware
from subspace.contrib.middleware.logging import LoggingMiddleware


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"


@tool
def get_time() -> str:
    """Get the current UTC time."""
    return datetime.now(UTC).isoformat()


memory = InMemorySaver()


def make_triage(ctx: RequestContext):
    return create_agent(
        "anthropic:claude-sonnet-4-6",
        system_prompt=(
            "You are a triage agent. You cannot answer questions yourself. "
            "Delegate weather questions to the 'weather' model "
            "and time questions to the 'time' model."
        ),
        tools=[],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"delegate_to": True})],
        checkpointer=memory,
    )


def make_weather(ctx: RequestContext):
    return create_agent(
        "anthropic:claude-sonnet-4-6",
        system_prompt="You are a weather assistant. Use the get_weather tool to answer.",
        tools=[get_weather],
    )


def make_time(ctx: RequestContext):
    return create_agent(
        "anthropic:claude-sonnet-4-6",
        system_prompt="You are a time assistant. Use the get_time tool to answer.",
        tools=[get_time],
    )


def main() -> None:
    mount = SubspaceMount(
        interfaces=[OpenResponsesInterface(prefix="/v1")],
    )

    mount.model(
        "weather",
        backend=LangchainBackend(make_weather),
        middlewares=[LoggingMiddleware()],
    )

    mount.model(
        "time",
        backend=LangchainBackend(make_time),
        middlewares=[LoggingMiddleware()],
    )

    mount.model(
        "triage",
        backend=LangchainBackend(make_triage),
        middlewares=[
            LoggingMiddleware(),
            DelegateMiddleware(mount.subspace, available_models=["weather", "time"]),
        ],
    )

    app = SubspaceApp(mount, title="Subspace Delegate Example")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
