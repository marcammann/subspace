"""Minimal MCP server with a single tool: get_altitude.

Run standalone:
    uv run python -m examples.mcp.server
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("altitude-server")

ALTITUDES = {
    "denver": 1609,
    "la paz": 3640,
    "kathmandu": 1400,
    "mexico city": 2240,
    "bogota": 2640,
    "quito": 2850,
    "addis ababa": 2355,
    "nairobi": 1795,
    "zurich": 408,
    "new york": 10,
    "london": 11,
    "tokyo": 40,
}


@mcp.tool()
def get_altitude(city: str) -> str:
    """Get the altitude of a city in meters above sea level."""
    altitude = ALTITUDES.get(city.lower())
    if altitude is None:
        return f"Unknown city: {city}"
    return f"{city} is at {altitude}m above sea level"


if __name__ == "__main__":
    mcp.run(transport="stdio")
