"""Streamlit chat UI for the Customer Service Data Analyst Agent.

Run with: streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.graph import compile_graph
from agent.memory import get_checkpointer
from agent.profile import get_profile_summary, update_profile_from_conversation


st.set_page_config(
    page_title="Customer Service Data Analyst",
    page_icon="📊",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    session_id = st.text_input("Session ID", value="default", help="Use the same ID to resume conversations.")
    user_id = st.text_input("User ID", value="default", help="Your identity for profile tracking.")

    st.divider()
    if st.button("📝 Show My Profile"):
        st.info(get_profile_summary(user_id))

    st.divider()
    st.markdown("### About")
    st.markdown(
        "This agent analyzes the **Bitext Customer Service** dataset. "
        "Ask about categories, intents, examples, distributions, or request summaries."
    )

# Main chat area
st.title("📊 Customer Service Data Analyst Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if "reasoning" in msg:
            with st.expander("🔍 Reasoning Steps"):
                for step in msg["reasoning"]:
                    st.markdown(step)

# Chat input
if prompt := st.chat_input("Ask about the customer service dataset..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            checkpointer = get_checkpointer()
            app = compile_graph(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": session_id}}
            input_state = {
                "messages": [HumanMessage(content=prompt)],
                "user_id": user_id,
            }

            # Collect reasoning steps
            reasoning_steps = []
            final_answer = ""

            for event in app.stream(input_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue

                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    reasoning_steps.append(
                                        f"🔧 **Tool call:** `{tc['name']}({tc['args']})`"
                                    )
                            elif msg.content:
                                final_answer = msg.content
                        elif isinstance(msg, ToolMessage):
                            content = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
                            reasoning_steps.append(f"📋 **Result:** {content}")

                    if "query_type" in node_output:
                        reasoning_steps.insert(
                            0, f"🔀 **Query type:** {node_output['query_type']}"
                        )

        # Display response
        st.markdown(final_answer)
        if reasoning_steps:
            with st.expander("🔍 Reasoning Steps"):
                for step in reasoning_steps:
                    st.markdown(step)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "reasoning": reasoning_steps,
    })

    # Update profile
    try:
        state = app.get_state(config)
        if state and state.values.get("messages"):
            update_profile_from_conversation(user_id, state.values["messages"])
    except Exception:
        pass
