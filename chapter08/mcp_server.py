"""Small stdio MCP server used by the Chapter 8 integration notebook."""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("chapter-8-travel-demo")


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """Return a sample flight option without making a booking."""
    return (
        f"Sample itinerary: {origin.upper()} to {destination.upper()} on {date}; "
        "nonstop, departing 09:00, arriving 17:30."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
