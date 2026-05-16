"""Memory management — checkpointer for persistent conversation state.

Uses SQLite locally and DynamoDB on AWS Lambda.
"""

import os
import sqlite3


_DB_DIR = os.path.dirname(os.path.dirname(__file__))
_DB_PATH = os.path.join(_DB_DIR, "conversations.db")
_DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "local")


def get_checkpointer():
    """Get a checkpointer for persistent conversation memory.

    Automatically selects backend based on DEPLOYMENT_MODE:
    - 'local' (default): SQLite file on disk.
    - 'aws_lambda': DynamoDB via custom saver.

    Returns:
        A LangGraph-compatible checkpointer instance.
    """
    if _DEPLOYMENT_MODE == "aws_lambda":
        from agent.dynamo_memory import DynamoDBSaver
        table_name = os.environ.get("CONVERSATIONS_TABLE", "agent-conversations")
        return DynamoDBSaver(table_name=table_name)

    # Local mode — SQLite
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
