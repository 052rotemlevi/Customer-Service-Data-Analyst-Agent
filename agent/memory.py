"""Memory management — SQLite-based checkpointer for persistent conversation state."""

import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


_DB_DIR = os.path.dirname(os.path.dirname(__file__))
_DB_PATH = os.path.join(_DB_DIR, "conversations.db")


def get_checkpointer() -> SqliteSaver:
    """Get a SQLite checkpointer for persistent conversation memory.

    Conversations survive across restarts. The same session ID will
    restore the full conversation history.

    Returns:
        SqliteSaver instance connected to the local SQLite database.
    """
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
