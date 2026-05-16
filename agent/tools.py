"""Tools for the Customer Service Data Analyst Agent.

Each tool has a clear name, description, and Pydantic input schema.
Tools are designed for composability — the agent can chain them for multi-step reasoning.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agent.dataset import get_dataframe


def _clean_null(value: Optional[str]) -> Optional[str]:
    """Convert LLM null-string values to Python None."""
    if value is None:
        return None
    if value.strip().lower() in ("null", "none", ""):
        return None
    return value


# ─────────────────────────────────────────────
# Pydantic Input Schemas
# ─────────────────────────────────────────────


class ListCategoriesInput(BaseModel):
    """No parameters needed."""
    pass


class ListIntentsInput(BaseModel):
    """Parameters for listing intents."""
    category: Optional[str] = Field(
        default=None,
        description="Optional category name to filter intents by (e.g., 'ACCOUNT', 'SHIPPING'). "
                    "If not provided, returns all unique intents across the dataset."
    )


class CountRowsInput(BaseModel):
    """Parameters for counting rows."""
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by (e.g., 'ACCOUNT', 'FEEDBACK')."
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent to filter by (e.g., 'get_refund', 'track_order'). "
                    "Use snake_case format."
    )


class GetExamplesInput(BaseModel):
    """Parameters for retrieving example rows."""
    n: int = Field(
        default=5,
        description="Number of examples to return (1-20).",
        ge=1,
        le=20,
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by."
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent to filter by (snake_case)."
    )


class GetDistributionInput(BaseModel):
    """Parameters for getting intent/category distribution."""
    group_by: str = Field(
        description="Column to group by: 'intent' or 'category'."
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by before computing distribution."
    )


class SearchExamplesInput(BaseModel):
    """Parameters for searching examples by keyword."""
    keyword: str = Field(
        description="Keyword or phrase to search for in the customer instruction text."
    )
    n: int = Field(
        default=5,
        description="Maximum number of matching examples to return.",
        ge=1,
        le=20,
    )


class SummarizeCategoryInput(BaseModel):
    """Parameters for summarizing a category or intent."""
    category: Optional[str] = Field(
        default=None,
        description="Category to summarize (e.g., 'FEEDBACK', 'ACCOUNT')."
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent to summarize (e.g., 'get_refund', 'cancel_order')."
    )


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────


@tool("list_categories", args_schema=ListCategoriesInput)
def list_categories() -> str:
    """List all unique categories in the Bitext Customer Service dataset.

    Use this tool to discover what top-level categories exist in the data.
    Returns a list of category names.
    """
    df = get_dataframe()
    categories = sorted(df["category"].dropna().unique().tolist())
    return f"Categories ({len(categories)}): {', '.join(categories)}"


@tool("list_intents", args_schema=ListIntentsInput)
def list_intents(category: Optional[str] = None) -> str:
    """List all unique intents in the dataset, optionally filtered by category.

    Use this tool to see what specific intent types exist.
    If a category is provided, only intents within that category are returned.
    """
    df = get_dataframe()
    category = _clean_null(category)
    if category:
        cat_upper = category.upper()
        df = df[df["category"].str.upper() == cat_upper]
        if df.empty:
            return f"No data found for category '{category}'. Use list_categories to see valid options."
    intents = sorted(df["intent"].dropna().unique().tolist())
    prefix = f"Intents in {category.upper()}" if category else "All intents"
    return f"{prefix} ({len(intents)}): {', '.join(intents)}"


@tool("count_rows", args_schema=CountRowsInput)
def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Count the number of rows in the dataset, optionally filtered by category and/or intent.

    Use this tool to get counts like 'How many refund requests?' or
    'How many entries in the SHIPPING category?'.
    Chain this after filter operations to get exact counts.
    """
    df = get_dataframe()
    category = _clean_null(category)
    intent = _clean_null(intent)
    filters_applied = []

    if category:
        cat_upper = category.upper()
        df = df[df["category"].str.upper() == cat_upper]
        filters_applied.append(f"category='{cat_upper}'")

    if intent:
        intent_lower = intent.lower()
        df = df[df["intent"].str.lower() == intent_lower]
        filters_applied.append(f"intent='{intent_lower}'")

    filter_desc = " AND ".join(filters_applied) if filters_applied else "no filters"
    return f"Count ({filter_desc}): {len(df)} rows"


@tool("get_examples", args_schema=GetExamplesInput)
def get_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Retrieve sample rows from the dataset.

    Use this tool when the user asks to 'show examples', 'give me samples',
    or wants to see actual data entries. Returns customer instructions and
    agent responses formatted as text.
    """
    df = get_dataframe()
    category = _clean_null(category)
    intent = _clean_null(intent)

    if category:
        cat_upper = category.upper()
        df = df[df["category"].str.upper() == cat_upper]

    if intent:
        intent_lower = intent.lower()
        df = df[df["intent"].str.lower() == intent_lower]

    if df.empty:
        return "No matching rows found. Check category/intent names using list_categories or list_intents."

    sample = df.sample(n=min(n, len(df)))
    lines = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"--- Example {i} ---\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Intent: {row.get('intent', 'N/A')}\n"
            f"Customer: {row.get('instruction', 'N/A')}\n"
            f"Agent Response: {row.get('response', 'N/A')}\n"
        )
    return "\n".join(lines)


@tool("get_distribution", args_schema=GetDistributionInput)
def get_distribution(group_by: str, category: Optional[str] = None) -> str:
    """Get the frequency distribution of intents or categories.

    Use this tool when the user asks about 'distribution', 'breakdown',
    or 'how many of each'. Shows counts and percentages per group.
    """
    df = get_dataframe()
    category = _clean_null(category)

    if category:
        cat_upper = category.upper()
        df = df[df["category"].str.upper() == cat_upper]
        if df.empty:
            return f"No data found for category '{category}'."

    col = group_by.lower()
    if col not in df.columns:
        return f"Invalid group_by column '{group_by}'. Use 'intent' or 'category'."

    counts = df[col].value_counts()
    total = counts.sum()
    lines = [f"Distribution by '{col}'" + (f" (within {category.upper()})" if category else "") + f" — Total: {total}"]
    for name, count in counts.items():
        pct = count / total * 100
        lines.append(f"  {name}: {count} ({pct:.1f}%)")
    return "\n".join(lines)


@tool("search_examples", args_schema=SearchExamplesInput)
def search_examples(keyword: str, n: int = 5) -> str:
    """Search for examples containing a keyword in the customer instruction text.

    Use this tool when the user describes what they're looking for in natural language,
    e.g., 'people wanting their money back' or 'complaints about delivery'.
    Searches the 'instruction' column for case-insensitive keyword matches.
    """
    df = get_dataframe()
    mask = df["instruction"].str.contains(keyword, case=False, na=False)
    matches = df[mask]

    if matches.empty:
        return f"No examples found containing '{keyword}'. Try different keywords."

    sample = matches.head(n)
    lines = [f"Found {len(matches)} total matches for '{keyword}'. Showing {len(sample)}:"]
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"\n--- Match {i} ---\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Intent: {row.get('intent', 'N/A')}\n"
            f"Customer: {row.get('instruction', 'N/A')}\n"
            f"Agent Response: {row.get('response', 'N/A')}\n"
        )
    return "\n".join(lines)


@tool("summarize_data", args_schema=SummarizeCategoryInput)
def summarize_data(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Get a statistical summary and sample of data for a category or intent.

    Use this tool for open-ended questions like 'Summarize the FEEDBACK category'
    or 'How do agents respond to cancellation requests?'.
    Returns: count, intent breakdown (if category given), and representative examples.
    """
    df = get_dataframe()
    category = _clean_null(category)
    intent = _clean_null(intent)

    if category:
        cat_upper = category.upper()
        df = df[df["category"].str.upper() == cat_upper]
        filter_name = f"category '{cat_upper}'"
    elif intent:
        intent_lower = intent.lower()
        df = df[df["intent"].str.lower() == intent_lower]
        filter_name = f"intent '{intent_lower}'"
    else:
        filter_name = "entire dataset"

    if df.empty:
        return f"No data found for {filter_name}."

    lines = [f"Summary of {filter_name}:", f"  Total rows: {len(df)}"]

    # Intent breakdown (if filtering by category)
    if category and "intent" in df.columns:
        intent_counts = df["intent"].value_counts().head(10)
        lines.append(f"  Top intents:")
        for name, count in intent_counts.items():
            lines.append(f"    - {name}: {count}")

    # Sample responses
    sample = df.sample(n=min(3, len(df)))
    lines.append(f"\n  Representative examples:")
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        lines.append(
            f"    [{i}] Intent: {row.get('intent', 'N/A')}\n"
            f"        Customer: {row.get('instruction', 'N/A')[:150]}\n"
            f"        Response: {row.get('response', 'N/A')[:200]}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Export all tools
# ─────────────────────────────────────────────

ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    get_examples,
    get_distribution,
    search_examples,
    summarize_data,
]
