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

import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from tools import ALL_TOOLS

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """You are ResearchBuddy, an expert research assistant.
You run on the user's own computer and have tools to search the web, read pages,
check Wikipedia, calculate and save reports.

How to work:
1. For any factual or current topic, FIRST use `web_search`.
2. Use `read_webpage` on the 1-3 most useful URLs to get real details.
   Use `search_wikipedia` for background and definitions.
3. Use `calculator` for every calculation (percentages, growth, totals). Never guess numbers.
4. Base your answer ONLY on what the tools returned. If sources disagree, say so.

When the user asks for a REPORT (or says "research ..."), write it in Markdown:
   # <Title>
   ## Summary          (3-4 sentences)
   ## Key Findings     (bullet points with concrete facts and numbers)
   ## Details          (a few short paragraphs)
   ## Sources          (the URLs you actually used)
then call `save_report` with the title and the full report, and tell the user
where it was saved.

For simple questions, answer briefly and cite the source URL.
"""


def build_model(model_name: str = DEFAULT_MODEL) -> ChatOllama:
    """Connect to the local Ollama server (default http://localhost:11434)."""
    return ChatOllama(
        model=model_name,
        temperature=0,   # factual, repeatable answers
        num_ctx=16384,   # bigger context window so web pages fit (Ollama's default is small)
    )


def build_agent(model=None, tools=None):
    """Create the LangChain agent. `model`/`tools` can be swapped (used by the tests)."""
    return create_agent(
        model=model or build_model(),
        tools=tools or ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # memory: remembers the chat per thread_id
    )
