"""Query router — classifies incoming queries as structured, unstructured, or out-of-scope.

Uses a lightweight LLM call to classify before the agent begins tool selection.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent.config import get_router_llm


ROUTER_SYSTEM_PROMPT = """You are a query classifier for a Customer Service Data Analyst Agent.
The agent has access ONLY to the Bitext Customer Service dataset which contains customer support queries and agent responses, categorized by 'category' and 'intent'.

Classify the user's query into exactly one of these types:

1. **structured** — Questions with concrete, data-driven answers that can be answered by filtering, counting, or retrieving data.
   Examples: "How many refund requests?", "What categories exist?", "Show me 3 examples from SHIPPING", "What is the distribution of intents?"

2. **unstructured** — Open-ended questions requiring summarization or analysis of the data content.
   Examples: "Summarize the FEEDBACK category", "How do agents respond to complaints?", "What patterns exist in refund requests?"

3. **out_of_scope** — Questions unrelated to the customer service dataset.
   Examples: "Who won the Champions League?", "Write me a poem", "What's the weather?", "What's the best CRM software?"

IMPORTANT: If conversation history is provided, consider the context. Follow-up requests like "show me more", "what about refunds?", "show me 3 more", or "and for the SHIPPING category?" are NOT out_of_scope — they reference the dataset through prior context. Classify them as structured or unstructured based on what the follow-up is asking for.

Respond with ONLY one word: structured, unstructured, or out_of_scope"""


def classify_query(query: str, conversation_context: str = "") -> str:
    """Classify a user query into structured, unstructured, or out_of_scope.

    Args:
        query: The user's input question.
        conversation_context: Optional recent conversation history for context.

    Returns:
        One of: 'structured', 'unstructured', 'out_of_scope'
    """
    llm = get_router_llm()

    user_content = query
    if conversation_context:
        user_content = f"Recent conversation:\n{conversation_context}\n\nNew query to classify: {query}"

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    classification = response.content.strip().lower().replace(".", "")

    # Normalize response
    if "out" in classification or "scope" in classification:
        return "out_of_scope"
    elif "unstructured" in classification:
        return "unstructured"
    elif "structured" in classification:
        return "structured"
    else:
        # Default to structured if unclear
        return "structured"
