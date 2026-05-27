"""Helpers for converting Subspace tool definitions to LangChain tools."""

from typing import Any

from pydantic import BaseModel

from subspace.models.tools import FunctionTool


class DeferredFunctionCall(BaseModel):
    """Payload passed to interrupt() when a tool is deferred for external execution."""

    name: str
    arguments: dict[str, Any]


def make_interrupt_tools(tools: list[FunctionTool]) -> list[Any]:
    """Convert Subspace FunctionTool definitions into LangChain tools that interrupt.

    Each tool calls langgraph's interrupt() when invoked, pausing the graph
    so the caller can provide the result externally (via Command(resume=...)).
    """
    from langchain_core.tools import StructuredTool
    from langgraph.types import interrupt

    result = []
    for tool_def in tools:

        def _make(td: FunctionTool):
            def _invoke(**kwargs: Any) -> str:
                payload = DeferredFunctionCall(name=td.name, arguments=kwargs)
                # Langchain is moving towards strict msg packing, so we use a dict here.
                return interrupt(payload.model_dump())

            return StructuredTool(
                name=td.name,
                description=td.description or "",
                args_schema=_json_schema_to_pydantic(td.name, td.parameters),
                func=_invoke,
            )

        result.append(_make(tool_def))

    return result


def _json_schema_to_pydantic(name: str, schema: dict[str, Any] | None) -> Any:
    """Convert a JSON Schema to a Pydantic model for use as args_schema."""
    from pydantic import create_model

    if not schema or not schema.get("properties"):
        return create_model(f"{name}_Args")

    properties = schema["properties"]
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_type_to_python(field_schema.get("type", "string"))
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type | None, None)

    return create_model(f"{name}_Args", **fields)


def _json_type_to_python(json_type: str) -> type:
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(json_type, str)
