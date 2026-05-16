"""User profile management — persistent per-user profiles.

Uses local JSON files by default, DynamoDB on AWS Lambda.
"""

import os
import json
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from agent.config import get_llm


_PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
_DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "local")


def _get_dynamo_table():
    """Get DynamoDB profiles table resource."""
    import boto3
    table_name = os.environ.get("PROFILES_TABLE", "agent-profiles")
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(table_name)


def _profile_path(user_id: str) -> str:
    """Get the file path for a user's profile."""
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    return os.path.join(_PROFILES_DIR, f"{safe_id}.json")


def load_profile(user_id: str) -> dict:
    """Load a user's profile.

    Args:
        user_id: Unique user identifier.

    Returns:
        Profile dictionary with keys like 'name', 'interests', 'preferences'.
    """
    default = {"user_id": user_id, "name": None, "facts": [], "interests": [], "preferences": []}

    if _DEPLOYMENT_MODE == "aws_lambda":
        try:
            table = _get_dynamo_table()
            response = table.get_item(Key={"user_id": user_id})
            item = response.get("Item")
            if item and "profile_data" in item:
                return json.loads(item["profile_data"])
        except Exception:
            pass
        return default

    # Local mode — JSON files
    path = _profile_path(user_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_profile(user_id: str, profile: dict) -> None:
    """Save a user's profile.

    Args:
        user_id: Unique user identifier.
        profile: Profile dictionary to persist.
    """
    if _DEPLOYMENT_MODE == "aws_lambda":
        try:
            table = _get_dynamo_table()
            table.put_item(Item={"user_id": user_id, "profile_data": json.dumps(profile)})
        except Exception:
            pass
        return

    # Local mode — JSON files
    path = _profile_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def update_profile_from_conversation(user_id: str, messages: list) -> dict:
    """Analyze recent conversation and update the user profile.

    Uses LLM to extract relevant facts about the user from the conversation.

    Args:
        user_id: Unique user identifier.
        messages: Recent conversation messages.

    Returns:
        Updated profile dictionary.
    """
    profile = load_profile(user_id)

    # Only process if there are enough messages
    if len(messages) < 2:
        return profile

    # Get recent messages (last 10)
    recent = messages[-10:]
    conversation_text = "\n".join(
        f"{'User' if msg.type == 'human' else 'Agent'}: {msg.content}"
        for msg in recent
        if hasattr(msg, "content") and msg.content and msg.type in ("human", "ai")
    )

    if not conversation_text.strip():
        return profile

    current_profile_str = json.dumps(profile, indent=2)

    llm = get_llm(temperature=0)
    extract_prompt = f"""Analyze this conversation and update the user profile.
Extract any new facts about the user: their name, topics they care about, preferences, or patterns.

Current profile:
{current_profile_str}

Recent conversation:
{conversation_text}

Return ONLY a valid JSON object with these fields:
- "user_id": "{user_id}"
- "name": string or null (user's name if mentioned)
- "facts": list of strings (distilled facts about the user)
- "interests": list of strings (topics they frequently ask about)
- "preferences": list of strings (their preferences or habits)

Keep existing information and add new facts. Do not duplicate. Return ONLY JSON, no explanation."""

    try:
        response = llm.invoke([HumanMessage(content=extract_prompt)])
        content = response.content.strip()
        # Try to parse JSON from the response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        updated = json.loads(content)
        updated["user_id"] = user_id
        save_profile(user_id, updated)
        return updated
    except (json.JSONDecodeError, Exception):
        # If parsing fails, keep existing profile
        return profile


def get_profile_summary(user_id: str) -> str:
    """Get a human-readable summary of the user's profile.

    Args:
        user_id: Unique user identifier.

    Returns:
        Formatted string describing what we know about the user.
    """
    profile = load_profile(user_id)

    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("facts"):
        parts.append(f"Facts: {'; '.join(profile['facts'])}")
    if profile.get("interests"):
        parts.append(f"Interests: {', '.join(profile['interests'])}")
    if profile.get("preferences"):
        parts.append(f"Preferences: {', '.join(profile['preferences'])}")

    if not parts:
        return "I don't have any information about you yet. As we chat, I'll remember your interests and preferences."

    return "Here's what I remember about you:\n" + "\n".join(f"  • {p}" for p in parts)
