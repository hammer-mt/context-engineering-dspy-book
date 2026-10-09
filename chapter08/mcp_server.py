"""Small stdio MCP server used by the Chapter 8 integration notebook.

Both tools return sample data. No real flight search or booking takes place.
"""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("chapter-8-travel-demo")


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between two airports on a date. Returns a sample option."""
    return (
        f"Sample itinerary: flight DEMO101, {origin.upper()} to "
        f"{destination.upper()} on {date}; nonstop, departing 09:00, "
        "arriving 17:30."
    )


@mcp.tool()
def book_flight(flight_id: str, passenger_name: str = "Sample Traveler") -> str:
    """Book a flight by the flight ID from search_flights. Simulated: no real booking is made."""
    return (
        f"Sample confirmation DEMO-0001: flight {flight_id.upper()} booked for "
        f"{passenger_name}. This is a simulated booking; no ticket was purchased."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
