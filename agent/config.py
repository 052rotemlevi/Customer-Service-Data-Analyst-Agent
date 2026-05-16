"""Configuration and LLM setup for the agent."""

import os
from langchain_openai import ChatOpenAI


# Nebius Token Factory API configuration
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

# Model choices:
# - Main agent: Llama-3.3-70B-Instruct (strong reasoning + native tool-calling support)
# - Router: Meta-Llama-3.1-8B-Instruct (fast, sufficient for classification)
AGENT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ROUTER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for Nebius AI Studio.

    Args:
        temperature: Sampling temperature (0 for deterministic).
        model: Model name override.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=model or AGENT_MODEL,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        temperature=temperature,
    )


def get_router_llm() -> ChatOpenAI:
    """Create a lightweight LLM for query routing (classification only).

    Returns:
        Configured ChatOpenAI instance using the smaller router model.
    """
    return ChatOpenAI(
        model=ROUTER_MODEL,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        temperature=0,
    )
