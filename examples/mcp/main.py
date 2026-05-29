"""MCP example: LiteLLM backend with an MCP server tool + stream observer.

Connects to a local MCP server (stdio) that provides a get_altitude tool.
A RetractionMiddleware runs in the background and retracts responses
that mention dangerously high altitudes.

Run:
    uv run python -m examples.mcp

Then:
    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "mcp-demo", "input": "What is the altitude of Denver?", "stream": true}'

    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "mcp-demo", "input": "What is the altitude of La Paz?", "stream": true}'
"""

import logging
import re

import uvicorn

from subspace import (
    LitellmBackend,
    McpMiddleware,
    OpenResponsesRouter,
    SubspaceApp,
    SubspaceMount,
)
from subspace.contrib.middleware.logging import LoggingMiddleware
from subspace.contrib.middleware.retraction import (
    ResponseOutputTextRetractedEvent,
    RetractionMiddleware,
)
from subspace.middleware.context import RequestContext
from subspace.models.events import ResponseOutputTextDeltaEvent, StreamEvent

logger = logging.getLogger("subspace.altitude_guard")


class AltitudeGuard(RetractionMiddleware):
    """Retracts responses that mention altitudes above 3000m."""

    @staticmethod
    def local_factory() -> dict:
        return {"text": "", "chars": 0}

    def should_dispatch(self, ctx: RequestContext, event: StreamEvent) -> bool:
        if not isinstance(event, ResponseOutputTextDeltaEvent):
            return False
        local = ctx.local(self)
        local["text"] += event.delta
        local["chars"] += len(event.delta)
        if local["chars"] >= 5:
            local["chars"] = 0
            return True
        return False

    async def consume(
        self, ctx: RequestContext, event: StreamEvent
    ) -> StreamEvent | list[StreamEvent] | None:
        text = ctx.local(self)["text"]
        match = re.search(r"([\d,]+)\s*m", text)
        if match:
            altitude = int(match.group(1).replace(",", ""))
            if altitude > 30:
                logger.warning("altitude %dm exceeds 3000m — retracting", altitude)
                return ResponseOutputTextRetractedEvent(
                    sequence_number=0,
                    reason=f"Altitude {altitude}m exceeds safety threshold of 3000m",
                )
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    mount = SubspaceMount(
        interfaces=[OpenResponsesRouter(prefix="/v1")],
    )
    mount.agent(
        "mcp-demo",
        backend=LitellmBackend(model="anthropic/claude-sonnet-4-6"),
        middlewares=[
            LoggingMiddleware(),
            AltitudeGuard(),
            McpMiddleware(
                command="uv",
                args=["run", "python", "-m", "examples.mcp.server"],
            ),
        ],
    )

    app = SubspaceApp(mount, title="Subspace MCP Example")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
