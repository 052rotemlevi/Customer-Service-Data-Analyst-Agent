"""CLI interface for the Customer Service Data Analyst Agent.

Usage:
    python main.py                     # Start with default session
    python main.py --session my_session  # Resume a specific session
    python main.py --user alice          # Set user identity for profile

The agent drops into an interactive loop, printing reasoning steps and final answers.
"""

import argparse
import sys

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.graph import compile_graph
from agent.memory import get_checkpointer
from agent.profile import get_profile_summary, update_profile_from_conversation
from agent.recommender import suggest_next_query, is_recommendation_request, is_confirmation


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text to the terminal."""
    colors = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def print_reasoning_steps(events: list) -> str:
    """Print the agent's reasoning steps and return the final answer."""
    final_answer = ""

    for event in events:
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            messages = node_output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    # Print tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print_colored(
                                f"  🔧 Tool call: {tc['name']}({tc['args']})",
                                "cyan"
                            )
                    # Print final answer
                    elif msg.content:
                        final_answer = msg.content

                elif isinstance(msg, ToolMessage):
                    # Print tool results (truncated)
                    content = msg.content[:300]
                    if len(msg.content) > 300:
                        content += "..."
                    print_colored(f"  📋 Result: {content}", "yellow")

            # Print routing info
            if "query_type" in node_output:
                qtype = node_output["query_type"]
                print_colored(f"  🔀 Query classified as: {qtype}", "magenta")

    return final_answer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="default",
        help="Session ID for conversation persistence (default: 'default')",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="User ID for profile management (default: 'default')",
    )
    args = parser.parse_args()

    session_id = args.session
    user_id = args.user

    # Initialize
    print_colored("=" * 60, "green")
    print_colored("  Customer Service Data Analyst Agent", "green")
    print_colored("=" * 60, "green")
    print_colored(f"  Session: {session_id} | User: {user_id}", "blue")
    print_colored(
        "  Type your questions about the Bitext Customer Service dataset.",
        "white",
    )
    print_colored("  Type 'quit' or 'exit' to end. Type 'profile' to see your profile.", "white")
    print_colored("=" * 60, "green")
    print()

    # Set up checkpointer and compile graph
    checkpointer = get_checkpointer()
    app = compile_graph(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    # Interactive loop
    pending_suggestion = None  # For query recommender (Bonus B)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print_colored("\nGoodbye! Your conversation has been saved.", "green")
            break

        if user_input.lower() == "profile":
            summary = get_profile_summary(user_id)
            print_colored(f"\n📝 {summary}", "blue")
            continue

        # Bonus B: Query Recommender
        if is_recommendation_request(user_input):
            state = app.get_state(config)
            all_msgs = state.values.get("messages", []) if state else []
            suggestion = suggest_next_query(all_msgs, user_id)
            pending_suggestion = suggestion
            print_colored(f"\n💡 Suggested query: \"{suggestion}\"", "cyan")
            print_colored("   Say 'yes' to execute, or refine your request.", "white")
            continue

        if pending_suggestion and is_confirmation(user_input):
            user_input = pending_suggestion
            print_colored(f"\n  Executing: \"{user_input}\"", "blue")
            pending_suggestion = None

        elif pending_suggestion:
            # User is refining — clear pending and treat as new query
            pending_suggestion = None

        # Run the agent
        print_colored("\n  Thinking...", "blue")

        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
        }

        try:
            # Stream events to show reasoning
            events = []
            for event in app.stream(input_state, config=config, stream_mode="updates"):
                events.append(event)

            final_answer = print_reasoning_steps(events)

            if final_answer:
                print_colored(f"\n🤖 Agent: {final_answer}", "green")

            # Update user profile periodically
            state = app.get_state(config)
            if state and state.values.get("messages"):
                all_messages = state.values["messages"]
                if len(all_messages) % 6 == 0:  # Every 3 exchanges
                    update_profile_from_conversation(user_id, all_messages)

        except Exception as e:
            print_colored(f"\n❌ Error: {e}", "red")
            print_colored("  Please try rephrasing your question.", "white")


if __name__ == "__main__":
    main()
