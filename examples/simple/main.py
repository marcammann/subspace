"""Minimal example: LangChain agent with a server-side tool.

Run:
    uv run python -m examples.simple

Then:
    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "agent", "input": "What time is it?", "stream": true}'
"""

import logging
from datetime import UTC, datetime
from typing import Annotated

import uvicorn
from pydantic import Field

from subspace import (
    FunctionCallMiddleware,
    OpenResponsesRouter,
    RequestContext,
    SubspaceApp,
    SubspaceMount,
    server_tool,
)
from subspace.contrib.backends.langchain import LangchainBackend
from subspace.contrib.middleware.logging import LoggingMiddleware

from .agent import make_agent


class TimeToolMiddleware(FunctionCallMiddleware):
    @server_tool(name="get_time", description="Get the current UTC time.")
    async def get_time(
        self,
        ctx: RequestContext,
        city: Annotated[str, Field(description="City to get the time for")],
    ) -> str:
        return datetime.now(UTC).isoformat()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    mount = SubspaceMount(
        interfaces=[OpenResponsesRouter(prefix="/v1")],
    )
    mount.agent(
        "agent",
        backend=LangchainBackend(make_agent),
        middlewares=[LoggingMiddleware(), TimeToolMiddleware()],
    )

    app = SubspaceApp(mount, title="Subspace Simple Example")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
