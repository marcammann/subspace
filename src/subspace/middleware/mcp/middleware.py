"""
MCP middleware — connects to an external MCP server and exposes its tools.

Discovers tools from the MCP server on connection, injects them into requests,
and handles tool calls server-side. Supports stdio, SSE, and streamable HTTP
transports. Tool results (including _meta for MCP Apps) are preserved.

Usage:
    from subspace.middleware.mcp import McpMiddleware

    # stdio transport
    mcp = McpMiddleware(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])

    # SSE transport
    mcp = McpMiddleware(url="http://localhost:3001/sse")

    # Streamable HTTP transport
    mcp = McpMiddleware(url="http://localhost:3001/mcp", transport=McpTransport.STREAMABLE_HTTP)

    # With tool filtering
    mcp = McpMiddleware(
        command="my-server",
        allowed_tools=["read_file", "write_file"],  # only expose these
        tool_prefix="fs",  # tools become "fs_read_file", "fs_write_file"
    )
"""

import json
import logging
from contextlib import AsyncExitStack
from enum import StrEnum
from typing import Any

import httpx

from subspace.middleware.context import RequestContext
from subspace.middleware.function_call import FunctionCallMiddleware
from subspace.models.items import FunctionCall
from subspace.models.tools import FunctionTool


class McpTransport(StrEnum):
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


logger = logging.getLogger(__name__)


class McpMiddleware(FunctionCallMiddleware):
    """Middleware that bridges an MCP server's tools into the Subspace pipeline.

    Connects to an MCP server (stdio, SSE, or streamable HTTP), discovers its
    tools, and exposes them to the model. Tool calls are intercepted, routed to
    the MCP server, and returned as ServerFunctionCall items.

    For MCP Apps, the full structured result (including _meta) is preserved in
    the tool output as JSON when the result contains metadata beyond plain text.
    """

    def __init__(
        self,
        *,
        # stdio transport
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        # HTTP transports (SSE or streamable HTTP)
        url: str | None = None,
        headers: dict[str, str] | None = None,
        transport: McpTransport | None = None,
        # Tool filtering
        allowed_tools: list[str] | None = None,
        excluded_tools: list[str] | None = None,
        tool_prefix: str = "",
        # Authentication (HTTP transports only)
        auth: httpx.Auth | None = None,
        # Client options
        timeout: float | None = None,
        client_name: str = "subspace",
        client_version: str = "1.0.0",
    ) -> None:
        if not command and not url:
            msg = "McpMiddleware requires either 'command' (stdio) or 'url' (SSE/HTTP)"
            raise ValueError(msg)

        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._url = url
        self._headers = headers
        self._transport = transport
        self._allowed_tools = set(allowed_tools) if allowed_tools else None
        self._excluded_tools = set(excluded_tools) if excluded_tools else None
        self._tool_prefix = tool_prefix
        self._auth = auth
        self._timeout = timeout
        self._client_name = client_name
        self._client_version = client_version

        self._session: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, FunctionTool] = {}
        self._tool_name_map: dict[str, str] = {}  # prefixed_name -> original_name
        self._connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection to the MCP server and discover tools."""
        from mcp import ClientSession
        from mcp.types import Implementation

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        read_stream, write_stream = await self._open_transport()

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                client_info=Implementation(name=self._client_name, version=self._client_version),
            )
        )
        await self._session.initialize()
        await self._refresh_tools()
        self._connected = True
        logger.info(
            "MCP connected: %d tools available",
            len(self._tools),
        )

    async def disconnect(self) -> None:
        """Close the MCP server connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        self._connected = False
        self._tools.clear()
        self._tool_name_map.clear()

    async def _open_transport(self) -> tuple[Any, Any]:
        """Open the appropriate transport and return (read_stream, write_stream)."""
        assert self._exit_stack is not None

        if self._command:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=self._env,
                cwd=self._cwd,
            )
            return await self._exit_stack.enter_async_context(stdio_client(params))

        assert self._url is not None
        if self._transport == McpTransport.STREAMABLE_HTTP:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

            http_client = create_mcp_http_client(
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout) if self._timeout else None,
                auth=self._auth,
            )
            await self._exit_stack.enter_async_context(http_client)
            read, write, *_ = await self._exit_stack.enter_async_context(
                streamable_http_client(url=self._url, http_client=http_client)
            )
            return read, write

        from mcp.client.sse import sse_client

        return await self._exit_stack.enter_async_context(
            sse_client(
                url=self._url,
                headers=self._headers or {},
                timeout=self._timeout or 5,
                auth=self._auth,
            )
        )

    async def _refresh_tools(self) -> None:
        """Fetch the tool list from the MCP server and apply filtering."""
        result = await self._session.list_tools()

        self._tools.clear()
        self._tool_name_map.clear()

        for tool in result.tools:
            if not self._should_include_tool(tool.name):
                continue

            prefixed_name = f"{self._tool_prefix}_{tool.name}" if self._tool_prefix else tool.name
            self._tool_name_map[prefixed_name] = tool.name

            self._tools[prefixed_name] = FunctionTool(
                name=prefixed_name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
            )

    def _should_include_tool(self, name: str) -> bool:
        if self._allowed_tools is not None:
            return name in self._allowed_tools
        if self._excluded_tools is not None:
            return name not in self._excluded_tools
        return True

    def owns_function(self, name: str) -> bool:
        return name in self._tools

    async def prepare(self, ctx: RequestContext) -> None:
        if not self._connected:
            await self.connect()

        if self._tools:
            existing = ctx.request.tools or []
            ctx.request = ctx.request.model_copy(
                update={"tools": existing + list(self._tools.values())}
            )

    async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
        original_name = self._tool_name_map[call.name]
        arguments = json.loads(call.arguments) if call.arguments else {}
        result = await self._session.call_tool(original_name, arguments)
        return _format_tool_result(result)

    @property
    def tools(self) -> dict[str, FunctionTool]:
        """The currently available MCP tools (after connection)."""
        return dict(self._tools)

    @property
    def connected(self) -> bool:
        return self._connected

    async def refresh_tools(self) -> None:
        """Re-fetch the tool list from the MCP server (e.g. after server notifies of changes)."""
        if not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)
        await self._refresh_tools()


def _format_tool_result(result: Any) -> str:
    """
    Format an MCP CallToolResult into a string for the model.

    For simple text-only results, returns the plain text.
    For results with metadata, errors, or mixed content, returns JSON
    so the model can see the full structured response (including _meta for MCP Apps).
    """
    from mcp.types import CallToolResult, TextContent

    if not isinstance(result, CallToolResult):
        return str(result)

    content_blocks = result.content
    meta = result.meta
    structured = result.structuredContent

    # Simple case: single text block, no error, no metadata
    if (
        len(content_blocks) == 1
        and isinstance(content_blocks[0], TextContent)
        and not result.isError
        and not meta
        and not structured
    ):
        return content_blocks[0].text

    # Rich result: serialize to JSON for full fidelity
    output: dict[str, Any] = {}

    if result.isError:
        output["is_error"] = True

    content_list = []
    for block in content_blocks:
        if isinstance(block, TextContent):
            content_list.append({"type": "text", "text": block.text})
        else:
            content_list.append({"type": block.type, "data": "[binary]"})
    output["content"] = content_list

    if structured:
        output["structured_content"] = structured

    if meta:
        output["_meta"] = meta

    return json.dumps(output)
