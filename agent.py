"""
The AGENT = LLM (brain) + Tools (hands) + Prompt (instructions) + Memory.

LangChain's `create_agent` builds a "ReAct" loop (Reason -> Act -> Observe):

    your question
         |
         v
    +-----------+   "I need a tool"   +-----------+
    |   MODEL   | ------------------> |   TOOLS   |
    | (Ollama)  | <------------------ | (Python)  |
    +-----------+    tool result      +-----------+
         |
         |  "I have enough information"
         v
    final answer
"""

import json
import os
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from tools import ALL_TOOLS

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")

SYSTEM_PROMPT = """You are ResearchBuddy, a research assistant with tools.

Rules:
- For facts or current topics, use `web_search` first, then `read_webpage` on the 1-2 best URLs.
- Use `search_wikipedia` for definitions and background.
- Use `calculator` for every calculation. Never guess numbers.
- Only use facts that the tools returned. Mention the source URLs.

If the user asks for a REPORT or says "research ...", write the final answer in Markdown:
# <Title>
## Summary
## Key Findings
## Details
## Sources
Then call `save_report` with the title and the full report.

For simple questions, answer briefly.
"""


def build_model(model_name: str = DEFAULT_MODEL) -> ChatOllama:
    """Connect to the local Ollama server (default http://localhost:11434)."""
    return ChatOllama(
        model=model_name,
        temperature=0,  # factual, repeatable answers
        num_ctx=8192,   # context window: big enough for a few web pages, light enough for a laptop
    )


# ---------------------------------------------------------------------------
# MIDDLEWARE: code that runs inside the agent loop, here right after each
# model call. Small local models sometimes write a tool call as plain text,
# e.g.  [{"name": "calculator", "arguments": {"expression": "2+2"}}]
# instead of a real tool call. This hook turns that text into a real call
# so the agent keeps working.
# ---------------------------------------------------------------------------
TOOL_NAMES = {t.name for t in ALL_TOOLS}


def parse_text_tool_calls(text: str) -> list[dict]:
    """Find a JSON tool call written inside plain text. Returns [] if there is none."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        calls = []
        for item in items:
            if not isinstance(item, dict) or item.get("name") not in TOOL_NAMES:
                break
            args = item.get("arguments", item.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    break
            if not isinstance(args, dict):
                break
            calls.append({"name": item["name"], "args": args,
                          "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
        else:
            if calls:
                return calls
    return []


@after_model
def fix_text_tool_calls(state, runtime):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls and last.text:
        calls = parse_text_tool_calls(last.text)
        if calls:
            # same message id -> LangChain replaces the message instead of adding one
            return {"messages": [AIMessage(content="", tool_calls=calls, id=last.id)]}
    return None


def build_agent(model=None, tools=None):
    """Create the LangChain agent. `model`/`tools` can be swapped (used by the tests)."""
    return create_agent(
        model=model or build_model(),
        tools=tools or ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[fix_text_tool_calls],
        checkpointer=InMemorySaver(),  # memory: remembers the chat per thread_id
    )
