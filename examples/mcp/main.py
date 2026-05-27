"""MCP example: LiteLLM backend with an MCP server tool.

Connects to a local MCP server (stdio) that provides a get_altitude tool.

Run:
    uv run python -m examples.mcp

Then:
    curl -N -X POST http://localhost:8001/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "mcp-demo", "input": "What is the altitude of Denver?", "stream": true}'
"""

import logging

import uvicorn

from subspace import (
    LitellmBackend,
    McpMiddleware,
    OpenResponsesInterface,
    SubspaceApp,
    SubspaceMount,
)
from subspace.contrib.middleware.logging import LoggingMiddleware


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    mount = SubspaceMount(
        interfaces=[OpenResponsesInterface(prefix="/v1")],
    )
    mount.model(
        "mcp-demo",
        backend=LitellmBackend(model="anthropic/claude-sonnet-4-6"),
        middlewares=[
            LoggingMiddleware(),
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
