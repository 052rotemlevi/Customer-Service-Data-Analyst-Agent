"""LangGraph ReAct agent with persistent memory and query routing."""

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.config import get_llm
from agent.router import classify_query
from agent.tools import ALL_TOOLS
from agent.profile import get_profile_summary, update_profile_from_conversation


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str  # structured, unstructured, out_of_scope
    iteration_count: int
    user_id: str


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MAX_ITERATIONS = 12

SYSTEM_PROMPT = """You are a Customer Service Data Analyst Agent. You help users analyze the Bitext Customer Service dataset.

The dataset contains customer support queries with these columns:
- category: Top-level category (e.g., ACCOUNT, SHIPPING, ORDER, FEEDBACK, etc.)
- intent: Specific intent (e.g., get_refund, track_order, cancel_order, etc.)
- instruction: The customer's message/query
- response: The agent's response to the customer

You have access to tools for querying and analyzing this dataset. Use them to answer user questions accurately.

IMPORTANT RULES:
- Only answer questions about the customer service dataset.
- If a question is out of scope, politely decline and explain you can only help with dataset analysis.
- Use tools to gather data, then respond to the user with a clear answer.
- NEVER call the same tool twice with the same arguments. Once you receive a tool result, use it to formulate your answer.
- After receiving tool results, synthesize the information and respond directly to the user in natural language.
- For counting questions, use count_rows with the right filters.
- For examples, use get_examples or search_examples.
- For summaries, use summarize_data.
- For distributions, use get_distribution.

{profile_info}"""


# ─────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify the incoming query before agent processes it."""
    messages = state["messages"]
    # Get the last human message
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if last_human is None:
        return {"query_type": "structured", "iteration_count": 0}

    # Build conversation context from prior messages for follow-up awareness
    context_parts = []
    for msg in messages[:-1]:  # Exclude the current message
        if isinstance(msg, HumanMessage):
            context_parts.append(f"User: {msg.content[:150]}")
        elif isinstance(msg, AIMessage) and msg.content:
            context_parts.append(f"Agent: {msg.content[:150]}")
    conversation_context = "\n".join(context_parts[-6:])  # Last few turns

    query_type = classify_query(last_human, conversation_context)
    return {"query_type": query_type, "iteration_count": 0}


def should_continue_after_router(state: AgentState) -> str:
    """Decide next node based on query classification."""
    if state["query_type"] == "out_of_scope":
        return "decline"
    return "agent"


def decline_node(state: AgentState) -> dict:
    """Politely decline out-of-scope queries."""
    decline_msg = AIMessage(
        content="I appreciate your question, but I can only help with analyzing the "
                "Bitext Customer Service dataset. I can help you explore categories, "
                "intents, view examples, get distributions, or summarize patterns in "
                "customer service interactions. What would you like to know about the dataset?"
    )
    return {"messages": [decline_msg]}


def agent_node(state: AgentState) -> dict:
    """Main agent reasoning node — decides whether to use tools or respond."""
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    user_id = state.get("user_id", "default")

    # Check max iterations
    if iteration_count >= MAX_ITERATIONS:
        fallback_msg = AIMessage(
            content="I've reached my maximum reasoning steps for this query. "
                    "Based on what I've gathered so far, let me provide what I can. "
                    "Could you try rephrasing your question or breaking it into smaller parts?"
        )
        return {"messages": [fallback_msg], "iteration_count": iteration_count + 1}

    # Detect tool-call loops: if the last tool result is the same tool called 2+ times, force a text answer
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if len(tool_messages) >= 2:
        last_two_names = [m.name for m in tool_messages[-2:]]
        if last_two_names[0] == last_two_names[1]:
            # Force the model to answer without tools
            profile_info = get_profile_summary(user_id)
            system_msg = SystemMessage(
                content=SYSTEM_PROMPT.format(profile_info=f"\nUser profile: {profile_info}")
                + "\n\nYou already have the tool results you need. Do NOT call any more tools. "
                  "Respond to the user now with a clear, helpful answer based on the tool results above."
            )
            llm = get_llm(temperature=0)
            llm_messages = [system_msg] + messages
            response = llm.invoke(llm_messages)
            return {"messages": [response], "iteration_count": iteration_count + 1}

    # Build system prompt with profile
    profile_info = get_profile_summary(user_id)
    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(profile_info=f"\nUser profile: {profile_info}")
    )

    # Prepare messages for the LLM
    llm_messages = [system_msg] + messages

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    response = llm_with_tools.invoke(llm_messages)

    return {"messages": [response], "iteration_count": iteration_count + 1}


def should_continue_after_agent(state: AgentState) -> str:
    """Decide whether to call tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)

    # If max iterations exceeded, end
    if iteration_count >= MAX_ITERATIONS:
        return END

    # If the LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, the agent has a final answer
    return END


def profile_update_node(state: AgentState) -> dict:
    """Update user profile based on conversation (runs at the end)."""
    user_id = state.get("user_id", "default")
    messages = state["messages"]
    update_profile_from_conversation(user_id, messages)
    return {}


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_graph():
    """Build the LangGraph agent graph.

    Graph structure:
        router → (out_of_scope) → decline → END
        router → (structured/unstructured) → agent ⇄ tools → END
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add edges
    graph.add_conditional_edges(
        "router",
        should_continue_after_router,
        {"decline": "decline", "agent": "agent"},
    )
    graph.add_edge("decline", END)
    graph.add_conditional_edges(
        "agent",
        should_continue_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with an optional checkpointer for memory persistence.

    Args:
        checkpointer: LangGraph checkpointer for conversation persistence.

    Returns:
        Compiled graph ready for invocation.
    """
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
