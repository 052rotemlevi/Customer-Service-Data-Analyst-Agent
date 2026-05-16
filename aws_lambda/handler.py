"""AWS Lambda handler for the Customer Service Data Analyst Agent.

Receives HTTP requests via API Gateway and returns agent responses.
Uses DynamoDB for conversation persistence and user profiles.
"""

import json
import os
import traceback
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Set deployment mode before importing agent modules
os.environ.setdefault("DEPLOYMENT_MODE", "aws_lambda")

from agent.graph import compile_graph
from agent.memory import get_checkpointer
from agent.profile import get_profile_summary


def lambda_handler(event: dict, context: Any) -> dict:
    """Lambda entry point for API Gateway requests.

    POST /chat — Send a query to the agent.
    Body: {"query": "...", "session_id": "...", "user_id": "..."}

    GET /health — Health check.
    """
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")

    # CORS headers
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }

    # Handle OPTIONS (CORS preflight)
    if http_method == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    # Health check
    if path in ("/health", "/api/health") or (http_method == "GET" and "/chat" not in path):
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({"status": "healthy", "service": "Customer Service Data Analyst Agent"}),
        }

    # POST /chat (or /api/chat via CloudFront)
    if http_method == "POST" and ("chat" in path):
        try:
            body = json.loads(event.get("body", "{}"))
            query = body.get("query", "").strip()
            session_id = body.get("session_id", "default")
            user_id = body.get("user_id", "default")

            if not query:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "Missing 'query' field in request body."}),
                }

            # Run the agent
            result = run_agent(query, session_id, user_id)

            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps(result),
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "statusCode": 500,
                "headers": headers,
                "body": json.dumps({"error": str(e)}),
            }

    return {
        "statusCode": 404,
        "headers": headers,
        "body": json.dumps({"error": "Not found. Use POST /chat or GET /health."}),
    }


def run_agent(query: str, session_id: str, user_id: str) -> dict:
    """Run the agent graph and return structured response.

    Args:
        query: User's question.
        session_id: Conversation session ID.
        user_id: User ID for profile.

    Returns:
        Dict with 'answer', 'reasoning_steps', and 'query_type'.
    """
    checkpointer = get_checkpointer()
    app = compile_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
    }

    reasoning_steps = []
    final_answer = ""
    query_type = ""

    for event in app.stream(input_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            messages = node_output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            reasoning_steps.append({
                                "type": "tool_call",
                                "tool": tc["name"],
                                "args": tc["args"],
                            })
                    elif msg.content:
                        final_answer = msg.content
                elif isinstance(msg, ToolMessage):
                    reasoning_steps.append({
                        "type": "tool_result",
                        "content": msg.content[:500],
                    })

            if "query_type" in node_output:
                query_type = node_output["query_type"]

    return {
        "answer": final_answer,
        "query_type": query_type,
        "reasoning_steps": reasoning_steps,
        "session_id": session_id,
        "user_id": user_id,
    }
