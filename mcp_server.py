"""FastMCP server exposing dataset analysis tools.

Exposes the agent's tools as MCP tools that any MCP-compatible client can call.
Start with: python mcp_server.py
"""

from typing import Optional
from pydantic import Field
from fastmcp import FastMCP

from agent.dataset import get_dataframe


# Create the MCP server
mcp = FastMCP(
    name="Customer Service Data Analyst",
    instructions=(
        "This server provides tools for analyzing the Bitext Customer Service dataset. "
        "Use these tools to query categories, intents, get examples, and analyze distributions."
    ),
)


@mcp.tool()
def list_categories() -> str:
    """List all unique categories in the Bitext Customer Service dataset.

    Returns a comma-separated list of all category names available in the data.
    """
    df = get_dataframe()
    categories = sorted(df["category"].dropna().unique().tolist())
    return f"Categories ({len(categories)}): {', '.join(categories)}"


@mcp.tool()
def count_rows(
    category: Optional[str] = Field(
        default=None,
        description="Category to filter by (e.g., 'ACCOUNT', 'SHIPPING')."
    ),
    intent: Optional[str] = Field(
        default=None,
        description="Intent to filter by (e.g., 'get_refund', 'track_order')."
    ),
) -> str:
    """Count the number of rows in the dataset, optionally filtered by category and/or intent.

    Use this to answer questions like 'How many refund requests are there?'
    or 'How many entries are in the SHIPPING category?'.
    """
    df = get_dataframe()
    filters = []

    if category:
        df = df[df["category"].str.upper() == category.upper()]
        filters.append(f"category='{category.upper()}'")

    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
        filters.append(f"intent='{intent.lower()}'")

    filter_desc = " AND ".join(filters) if filters else "no filters"
    return f"Count ({filter_desc}): {len(df)} rows"


@mcp.tool()
def get_examples(
    n: int = Field(default=5, description="Number of examples (1-20).", ge=1, le=20),
    category: Optional[str] = Field(default=None, description="Category filter."),
    intent: Optional[str] = Field(default=None, description="Intent filter."),
) -> str:
    """Retrieve sample rows from the dataset.

    Returns customer messages and agent responses, optionally filtered
    by category and/or intent.
    """
    df = get_dataframe()

    if category:
        df = df[df["category"].str.upper() == category.upper()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]

    if df.empty:
        return "No matching rows found."

    sample = df.sample(n=min(n, len(df)))
    lines = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"--- Example {i} ---\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Intent: {row.get('intent', 'N/A')}\n"
            f"Customer: {row.get('instruction', 'N/A')}\n"
            f"Agent: {row.get('response', 'N/A')}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def get_distribution(
    group_by: str = Field(description="Column to group by: 'intent' or 'category'."),
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by before computing distribution."
    ),
) -> str:
    """Get the frequency distribution of intents or categories in the dataset.

    Shows counts and percentages per group. Use for questions about
    'distribution', 'breakdown', or 'how many of each type'.
    """
    df = get_dataframe()

    if category:
        df = df[df["category"].str.upper() == category.upper()]
        if df.empty:
            return f"No data for category '{category}'."

    col = group_by.lower()
    if col not in df.columns:
        return f"Invalid column '{group_by}'. Use 'intent' or 'category'."

    counts = df[col].value_counts()
    total = counts.sum()
    lines = [f"Distribution by '{col}'" + (f" (in {category.upper()})" if category else "") + f" — Total: {total}"]
    for name, count in counts.items():
        pct = count / total * 100
        lines.append(f"  {name}: {count} ({pct:.1f}%)")
    return "\n".join(lines)


@mcp.tool()
def search_examples(
    keyword: str = Field(description="Keyword to search for in customer messages."),
    n: int = Field(default=5, description="Max results to return.", ge=1, le=20),
) -> str:
    """Search for examples containing a keyword in the customer instruction text.

    Case-insensitive search across all customer messages in the dataset.
    """
    df = get_dataframe()
    mask = df["instruction"].str.contains(keyword, case=False, na=False)
    matches = df[mask]

    if matches.empty:
        return f"No examples found containing '{keyword}'."

    sample = matches.head(n)
    lines = [f"Found {len(matches)} matches for '{keyword}'. Showing {len(sample)}:"]
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"\n--- Match {i} ---\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Intent: {row.get('intent', 'N/A')}\n"
            f"Customer: {row.get('instruction', 'N/A')}\n"
            f"Agent: {row.get('response', 'N/A')}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
