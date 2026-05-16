# Assignment 3 - Customer Service Data Analyst Agent

**By: Rotem Levi & Yaniv Maymon**

A LangGraph-based ReAct agent that analyzes the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). The agent answers structured queries (counts, distributions, examples), open-ended questions (summaries, pattern analysis), and gracefully declines out-of-scope questions.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────┐     ┌───────────────┐
│  User Input │ ──▶ │  Router  │ ──▶ │ Out-of-Scope  │ ──▶ Decline
└─────────────┘     │  (LLM)   │     └───────────────┘
                    └──────────┘
                         │ structured / unstructured
                         ▼
                    ┌──────────┐     ┌───────────┐
                    │  Agent   │ ◀─▶ │   Tools   │
                    │  (LLM)   │     │ (7 tools) │
                    └──────────┘     └───────────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Answer  │
                    └──────────┘
```

### Model Choice

We use two models via Nebius AI Studio:

- **Agent (reasoning + tool use):** `meta-llama/Llama-3.3-70B-Instruct`
- **Router (classification):** `meta-llama/Meta-Llama-3.1-8B-Instruct`

**Justification:**
- Llama-3.3-70B-Instruct has native tool-calling support, strong reasoning, and avoids looping
- Llama-3.1-8B-Instruct is sufficient for simple classification (structured/unstructured/out-of-scope) and much faster
- Both run efficiently on Nebius infrastructure
- Splitting models optimizes latency without sacrificing quality

### Tools Defined

| Tool | Purpose |
|------|---------|
| `list_categories` | Lists all dataset categories |
| `list_intents` | Lists intents (optionally by category) |
| `count_rows` | Counts rows with category/intent filters |
| `get_examples` | Retrieves sample rows |
| `get_distribution` | Shows frequency distribution |
| `search_examples` | Keyword search in customer messages |
| `summarize_data` | Statistical summary with samples |

---

## Setup

### Prerequisites

- Python 3.11+
- Nebius API key (set as environment variable)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Assignment 3 - Customer Service Data Analyst Agent - Rotem Levi and Yaniv Maymon"

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set your Nebius API key
set NEBIUS_API_KEY=your_key_here          # Windows cmd
# export NEBIUS_API_KEY=your_key_here     # Linux/Mac
# Or copy .env.example to .env and fill in your key
```

The Bitext dataset is downloaded automatically on first run and cached in `data/`.

---

## How to Run

### CLI (Interactive)

```bash
# Default session
python main.py

# With specific session ID (persists across restarts)
python main.py --session my_analysis

# With user identity for profile tracking
python main.py --session work --user alice
```

The CLI prints reasoning steps (tool calls and results) alongside final answers.

### Streamlit UI (Bonus A)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Set session/user IDs in the sidebar.

---

## MCP Server (Task 3)

### Starting the Server

```bash
python mcp_server.py
```

This starts a FastMCP server exposing 5 tools: `list_categories`, `count_rows`, `get_examples`, `get_distribution`, `search_examples`.

### Connecting a Client

```python
from fastmcp import Client

async def main():
    async with Client("mcp_server.py") as client:
        # List available tools
        tools = await client.list_tools()
        print([t.name for t in tools])

        # Call a tool
        result = await client.call_tool(
            "count_rows",
            {"category": "ACCOUNT", "intent": "get_refund"}
        )
        print(result)

import asyncio
asyncio.run(main())
```

Or use the MCP CLI inspector:
```bash
fastmcp dev mcp_server.py
```

---

## Features

### Task 1 — ReAct Agent (50 pts)
- ✅ Query router (structured / unstructured / out-of-scope)
- ✅ 7 tools with Pydantic schemas and clear descriptions
- ✅ Multi-step reasoning (e.g., filter → count)
- ✅ CLI with reasoning output
- ✅ Max iterations fallback (12 iterations) with loop detection

### Task 2a — Conversation Memory (20 pts)
- ✅ SQLite-based checkpointer (persists across restarts)
- ✅ Session ID argument (`--session`)
- ✅ Follow-up query support ("Show 3 more", "What about refunds?")

### Task 2b — User Profile (10 pts)
- ✅ Persistent per-user JSON profiles in `profiles/`
- ✅ Extracts name, interests, preferences from conversation
- ✅ "What do you remember about me?" support via `profile` command
- ✅ Survives restarts

### Task 3 — MCP Server (20 pts)
- ✅ FastMCP server with 5 exposed tools
- ✅ Pydantic-validated inputs
- ✅ Client connection example above

### Bonus A — Streamlit UI (+10 pts)
- ✅ Chat interface with message history
- ✅ Reasoning steps in expandable sections
- ✅ Session/user ID in sidebar

### Bonus B — Query Recommender (+10 pts)
- ✅ "What should I query next?" triggers recommendation
- ✅ User can refine before confirming
- ✅ Only executes on explicit confirmation

---

## Example Interactions

```
🧑 You: What categories exist in the dataset?
  🔀 Query classified as: structured
  🔧 Tool call: list_categories({})
  📋 Result: Categories (11): ACCOUNT, CANCEL, ...
🤖 Agent: The dataset contains 11 categories: ACCOUNT, CANCEL, ...

🧑 You: How many refund requests did we get?
  🔀 Query classified as: structured
  🔧 Tool call: count_rows({"intent": "get_refund"})
  📋 Result: Count (intent='get_refund'): 228 rows
🤖 Agent: There are 228 refund requests in the dataset.

🧑 You: Who won the Champions League?
  🔀 Query classified as: out_of_scope
🤖 Agent: I can only help with analyzing the Bitext Customer Service dataset...
```

---

## Project Structure

```
├── main.py              # CLI entry point
├── app.py               # Streamlit UI (Bonus A)
├── mcp_server.py        # FastMCP server (Task 3)
├── requirements.txt     # Dependencies with versions
├── agent/
│   ├── __init__.py
│   ├── config.py        # LLM configuration (model, API keys)
│   ├── dataset.py       # Dataset loading and caching
│   ├── graph.py         # LangGraph ReAct graph definition
│   ├── memory.py        # SQLite checkpointer setup
│   ├── profile.py       # User profile management
│   ├── recommender.py   # Query recommender (Bonus B)
│   ├── router.py        # Query classification node
│   └── tools.py         # Tool definitions with Pydantic schemas
├── data/                # Cached dataset (auto-downloaded)
├── profiles/            # Per-user JSON profiles
└── conversations.db     # SQLite conversation persistence (auto-created)
```
