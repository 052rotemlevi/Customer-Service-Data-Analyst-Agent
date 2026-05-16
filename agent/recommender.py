"""Query recommender — suggests relevant follow-up queries based on conversation history and user profile.

Implements Bonus B: The agent suggests queries, lets the user refine, and only executes on confirmation.
"""

from langchain_core.messages import BaseMessage, HumanMessage
from agent.config import get_llm
from agent.profile import load_profile


def suggest_next_query(messages: list[BaseMessage], user_id: str) -> str:
    """Suggest a relevant follow-up query based on conversation history and user profile.

    Args:
        messages: Full conversation history.
        user_id: User ID for profile lookup.

    Returns:
        A suggested query string.
    """
    profile = load_profile(user_id)

    # Build context from recent messages
    recent = messages[-10:]
    conversation_context = "\n".join(
        f"{'User' if msg.type == 'human' else 'Agent'}: {msg.content[:200]}"
        for msg in recent
        if hasattr(msg, "content") and msg.content and msg.type in ("human", "ai")
    )

    interests = ", ".join(profile.get("interests", [])) or "none yet"

    llm = get_llm(temperature=0.7)
    prompt = f"""Based on this conversation about the Bitext Customer Service dataset, suggest ONE relevant follow-up query the user might want to ask next.

User's interests: {interests}

Recent conversation:
{conversation_context}

The dataset has these features: category, intent, instruction (customer message), response (agent reply).
Available analyses: list categories, list intents, count rows, get examples, get distribution, search by keyword, summarize.

Suggest a specific, actionable query. Be concise — just the suggested question, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().strip('"')


def is_recommendation_request(query: str) -> bool:
    """Check if the user is asking for a query recommendation.

    Args:
        query: User's input.

    Returns:
        True if the user wants a suggestion.
    """
    triggers = [
        "what should i query",
        "what should i ask",
        "suggest a query",
        "recommend a query",
        "what else can i ask",
        "what next",
        "suggest something",
        "what can i explore",
    ]
    query_lower = query.lower().strip()
    return any(trigger in query_lower for trigger in triggers)


def is_confirmation(query: str) -> bool:
    """Check if the user is confirming a suggested query.

    Args:
        query: User's input.

    Returns:
        True if the user is saying yes/confirming.
    """
    confirmations = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "do it",
        "go ahead", "execute", "run it", "confirm", "yes please",
    ]
    query_lower = query.lower().strip().rstrip("!.")
    return query_lower in confirmations
