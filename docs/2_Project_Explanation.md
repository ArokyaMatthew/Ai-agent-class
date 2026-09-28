# Project Explanation: ResearchBuddy (a LangChain AI Agent)

## 1. What the project is

**ResearchBuddy** is an **autonomous AI research agent** built with the **LangChain**
framework. It runs **fully on a local laptop** using the **phi4-mini** language model
through **Ollama**, so it needs no API key, no cloud service and no payment.

**Problem it solves:** a plain LLM answers only from memory. Its knowledge is outdated,
it cannot open websites, it makes arithmetic mistakes and it may invent facts. This
project wraps the same LLM in a LangChain **agent** and gives it **tools**, so it can
**search the internet, read real web pages, check Wikipedia, calculate exactly and
save a research report to disk**. The LLM decides by itself which tools to use and in
what order.

**Example:**
```
You: research solar energy in India and write a report
```
The agent then (1) searches the web, (2) reads the best pages, (3) calculates growth
percentages, (4) writes a structured Markdown report with sources and (5) saves it
to `reports/20260928-101500-solar-energy-in-india.md`.

---

## 2. Technology Stack

| Layer | Technology | Purpose in this project |
|---|---|---|
| Language | Python 3.10+ | The whole project |
| **AI framework** | **LangChain 1.x** (`langchain`, `langchain-core`) | Agent, tools, messages, middleware |
| **Agent runtime** | **LangGraph** (`langgraph`) | Runs the agent loop; provides memory (checkpointer) |
| **LLM integration** | **langchain-ollama** (`ChatOllama`) | Connects LangChain to the local Ollama model |
| **LLM** | **phi4-mini** (Microsoft, 3.8 billion parameters) through **Ollama** | The agent's "brain" |
| Web search | `ddgs` (DuckDuckGo Search) | Internet search without an API key |
| HTML parsing | `beautifulsoup4` | Extracts readable text from web pages |
| HTTP | `urllib` (Python standard library) | Downloads web pages and calls the Wikipedia API |
| Terminal UI | `rich` | Coloured panels, spinner, Markdown rendering |
| Testing | `pytest` + LangChain's `GenericFakeChatModel` | Offline automated tests |

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  main.py  —  User Interface (terminal)                                 │
│  reads input → agent.stream(...) → prints Action / Observation / Answer│
└───────────────────────────────┬────────────────────────────────────────┘
                                │ {"messages": [user question]}, thread_id
┌───────────────────────────────▼────────────────────────────────────────┐
│  agent.py  —  LangChain Agent  (create_agent → LangGraph graph)        │
│                                                                        │
│   System Prompt ──┐                                                    │
│                   ▼                                                    │
│   START ──► [ model node ] ──► [ after_model middleware ] ──┐          │
│               ChatOllama         fix_text_tool_calls        │          │
│               (phi4-mini)                                   │          │
│                   ▲                     tool_calls? ──yes──►[ tools ]  │
│                   │                          │               node      │
│                   └──────── ToolMessage ◄────┼───────────────┘         │
│                                              no                        │
│                                              ▼                         │
│                                             END  (final answer)        │
│                                                                        │
│   Memory: InMemorySaver checkpointer (conversation per thread_id)      │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ calls Python functions
┌───────────────────────────────▼────────────────────────────────────────┐
│  tools.py  —  6 LangChain Tools (@tool)                                │
│  web_search │ read_webpage │ search_wikipedia │ calculator │           │
│  get_today_date │ save_report                                          │
└───────┬──────────────┬───────────────┬───────────────────────┬─────────┘
        ▼              ▼               ▼                       ▼
   DuckDuckGo     Any website     Wikipedia API        reports/*.md (disk)
                                                                        
   Ollama server (localhost:11434) ◄── ChatOllama sends every model call here
```

---

## 4. Project Files

| File | What it contains |
|---|---|
| `agent.py` | Builds the agent: the LLM (`ChatOllama`), the system prompt, the middleware, memory, and `create_agent(...)` |
| `tools.py` | The 6 tools, each a Python function with the `@tool` decorator |
| `main.py` | Terminal app: chat loop, streams and prints every agent step, auto-saves reports |
| `tests/test_agent.py` | 8 automated tests that run the real agent loop with a fake LLM (no internet or Ollama needed) |
| `requirements.txt` | Python packages |
| `reports/` | Created automatically; holds the saved research reports |

---

## 5. The Tools (tools.py)

### How a function becomes a LangChain tool

```python
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression exactly, e.g. "(1450 - 1200) / 1200 * 100".
    Always use this for percentages, growth rates, averages or any arithmetic. ..."""
```

The `@tool` decorator (from `langchain.tools`) turns the function into a `BaseTool`
object. LangChain builds a **JSON schema** from the function name, the docstring and
the type hints. This is exactly what is sent to the LLM for the calculator:

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Calculate a math expression exactly, e.g. \"(1450 - 1200) / 1200 * 100\". Always use this for percentages, growth rates, ...",
    "parameters": {
      "type": "object",
      "properties": { "expression": { "type": "string" } },
      "required": ["expression"]
    }
  }
}
```

The **docstring is the instruction to the LLM**. It tells the model *when* to use
the tool. The LLM never runs the Python code; it only returns a request such as
`calculator(expression="45999*0.18")`, and LangChain runs the function.

**Error handling design:** every tool catches its own exceptions and **returns an error
message as text** instead of crashing. The LLM reads the error and can try something
else, for example another URL. This makes the agent robust.

### The 6 tools in detail

| # | Tool | Input | What it does internally | Output to LLM |
|---|---|---|---|---|
| 1 | `web_search` | `query` | Calls `DDGS().text(query, max_results=5)` (DuckDuckGo) | Top 5 results: title, URL, snippet |
| 2 | `read_webpage` | `url` | Downloads the page with `urllib` (browser User-Agent). **BeautifulSoup** removes `script, style, nav, header, footer, aside, form, noscript` and extracts the text, which is cut to 3000 characters so it fits the context window | Clean main text of the page |
| 3 | `search_wikipedia` | `topic` | Two HTTP calls: (1) the Wikipedia **search API** finds the best article title, (2) the **REST summary API** gets its summary | Article title, URL and summary |
| 4 | `calculator` | `expression` | **Safe evaluation:** `ast.parse` builds a syntax tree and `_evaluate` walks it, allowing only numbers, `+ - * / % **`, and `sqrt, log, log10, abs, round`. **Python's `eval()` is never used**, because running code written by an LLM is a security risk | e.g. `45999*0.18 = 8279.82` |
| 5 | `get_today_date` | none | `datetime.now()` | e.g. `Monday, 28 September 2026` |
| 6 | `save_report` | `title`, `markdown_content` | Makes a safe file name from the title plus a timestamp and writes the Markdown to `reports/` | `Report saved to reports/....md` |

Tools 1–3 give the agent **up-to-date knowledge** (perception), tool 4 gives it
**exact reasoning with numbers**, tool 5 gives it **awareness of time**, and tool 6
lets it **act on the real world** by writing a file.

---

## 6. LangChain Features Used (agent.py and main.py)

### 6.1 Chat Model Integration: `ChatOllama` (package `langchain-ollama`)
```python
ChatOllama(model="phi4-mini", temperature=0, num_ctx=8192)
```
- `ChatOllama` implements LangChain's standard **`BaseChatModel` interface**, so the
  rest of the code does not depend on Ollama. Swapping to another model means changing
  one string (`--model qwen2.5:7b`), or one class for a cloud model (`ChatOpenAI`).
- `temperature=0` gives factual, repeatable answers.
- `num_ctx=8192` sets the context window. Ollama's default is small, and web pages
  plus tool results need more room.
- It sends requests to the **Ollama server** on `http://localhost:11434`.
- It supports **tool calling**: `create_agent` automatically calls `model.bind_tools(tools)`.

### 6.2 Tools: `@tool` decorator (`langchain.tools`)
This is explained in section 5. `ALL_TOOLS` is the list handed to the agent.

### 6.3 System Prompt
`SYSTEM_PROMPT` in `agent.py` is sent as a **SystemMessage** before every model call. It defines:
- the **role**: "ResearchBuddy, a research assistant";
- the **strategy**: search first, then read 1–2 pages, use Wikipedia for background, use the calculator for every number;
- the **grounding rule**: use only facts from the tools and cite URLs, which reduces hallucination;
- the **output format**: a report with Summary / Key Findings / Details / Sources, then call `save_report`.

This is **prompt engineering**: the agent's behaviour is controlled mainly through this text.

### 6.4 The Agent: `create_agent` (`langchain.agents`)
```python
create_agent(
    model=build_model(),
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[fix_text_tool_calls],
    checkpointer=InMemorySaver(),
)
```
`create_agent` is LangChain v1's standard agent. It builds and compiles a **LangGraph
`StateGraph`** with this structure (checked in code: its nodes are `model` and `tools`):

- **State:** a list of `messages`.
- **`model` node:** sends the system prompt, all messages and the tool schemas to phi4-mini.
- **Conditional edge:** if the `AIMessage` has `tool_calls`, go to the `tools` node; otherwise go to `END`.
- **`tools` node:** runs the requested Python functions and appends one `ToolMessage` per call.
- **Edge:** `tools` → `model` (loop back).

This is the **ReAct (Reason + Act) pattern**. The loop repeats until the model answers
without calling a tool. `recursion_limit: 40` in `main.py` stops an endless loop.

### 6.5 Messages (`langchain_core.messages`)

| Message | In this project |
|---|---|
| `SystemMessage` | The system prompt |
| `HumanMessage` | The user's question |
| `AIMessage` | phi4-mini's reply: either `tool_calls` or the final answer |
| `ToolMessage` | Each tool's result, linked to its call by `tool_call_id` |

`main.py` checks each message's type to print `>> Action:` (an AIMessage with tool
calls), `Observation:` (a ToolMessage) or the final answer panel (an AIMessage with text).

### 6.6 Middleware: `@after_model` (`langchain.agents.middleware`)
Middleware is code that runs **inside the agent loop**. This project uses an
**after_model** hook called `fix_text_tool_calls`:
- **Problem:** small local models like phi4-mini sometimes write a tool call as plain text,
  e.g. `[{"name": "calculator", "arguments": {"expression": "2+2"}}]`, instead of a proper tool call.
  The agent would then stop and show that JSON to the user.
- **Solution:** after each model call, `parse_text_tool_calls` scans the text with a JSON decoder.
  If it finds a call to a **known tool name**, it builds a real `tool_calls` entry.
  The middleware returns a new `AIMessage` with the **same message id**, so LangGraph's
  `add_messages` reducer **replaces** the broken message and the loop continues to the tools node.
- Unknown tool names are ignored, which is a safety check.

### 6.7 Memory: `InMemorySaver` checkpointer (`langgraph.checkpoint.memory`)
- After each step the checkpointer **saves the graph state** (all messages) under a
  **`thread_id`**.
- `main.py` creates one `thread_id` (a UUID) per conversation and passes it in
  `config={"configurable": {"thread_id": ...}}`.
- On the next question LangGraph **loads the earlier messages**, so the agent remembers
  context, e.g. "My name is Priya" … "What is my name?".
- `/new` creates a new `thread_id`, which starts a fresh conversation.
- It is *in-memory*, so the history is cleared when the program exits. For persistence,
  `SqliteSaver` could replace it.

### 6.8 Streaming: `agent.stream(..., stream_mode="updates")`
- Instead of waiting for the final result (`invoke`), `stream` returns **each graph
  step as it finishes**, for example `{"model": {"messages": [...]}}` followed by `{"tools": {"messages": [...]}}`.
- This is how the terminal shows the agent's reasoning **live**, which makes the ReAct loop visible during a demo.

### 6.9 Testing with a fake LLM (`GenericFakeChatModel`)
- `tests/test_agent.py` subclasses LangChain's `GenericFakeChatModel` to create a
  **scripted LLM** that returns fixed `AIMessage`s, including tool calls.
- `monkeypatch` replaces DuckDuckGo and the web with fake data.
- This tests the **real** `create_agent` loop, tools, middleware and memory
  **offline and deterministically**.

**The 8 tests:**
1. The calculator is correct and blocks code injection (`__import__('os')` → Error).
2. Web search and page reading work, and `<script>` content is removed.
3. `save_report` creates the file.
4. The full research loop runs: search → read → calculate → save → answer.
5. Memory keeps both turns in the same thread.
6. The text tool-call parser works and ignores unknown tools.
7. The middleware repairs a text tool call inside the real agent.
8. A report is auto-saved even if the model forgets to call `save_report`.

---

## 7. Step-by-Step Execution of One Request

**Input:** `what is 18% GST on 45,999 rupees?`

| Step | Graph node | Message added to state | Printed on screen |
|---|---|---|---|
| 1 | (input) | `HumanMessage("what is 18% GST on 45,999 rupees?")` | — |
| 2 | `model` | LLM receives the system prompt, the question and 6 tool schemas, and decides it needs maths: `AIMessage(tool_calls=[calculator(expression="45999*0.18")])` | `>> Action: calculator(expression='45999*0.18')` |
| 3 | `tools` | Runs `calculator()`: `ToolMessage("45999*0.18 = 8279.82")` | `Observation: 45999*0.18 = 8279.82` |
| 4 | `model` | LLM sees the result and has enough information: `AIMessage("18% GST on ₹45,999 is ₹8,279.82")` with no tool calls | Green answer panel |
| 5 | `END` | Checkpointer saves all 4 messages under the `thread_id` | — |

A **research request** follows the same loop with more iterations:
`web_search` → `read_webpage` (1–2 times) → `search_wikipedia` → `calculator` →
final Markdown report → `save_report`.

---

## 8. main.py: the User Interface

1. `argparse` reads an optional question and `--model` (default `phi4-mini`).
2. `build_agent(model=build_model(args.model))` creates the agent.
3. **Interactive loop:** it reads input; `/new` resets memory and `/exit` quits.
4. `run_turn()` streams the agent's steps and prints:
   - yellow `>> Action: tool(args)` for each tool call;
   - grey `Observation: ...`, a preview of each tool result;
   - a green panel with the final answer rendered as Markdown (using `rich`).
5. **Safety net:** if the final answer is a report (it starts with `# `) but the model
   did not call `save_report`, `main.py` saves it itself.
6. **Friendly errors:** if Ollama is not running or the model is missing, it prints how to fix it.

---

## 9. LangChain Concepts: Used vs Not Used

| LangChain concept | Used? | Where / note |
|---|---|---|
| Chat model integration | Yes | `ChatOllama` |
| Messages | Yes | System, Human, AI and Tool messages |
| Tools (`@tool`) | Yes | 6 tools in `tools.py` |
| Tool calling (`bind_tools`) | Yes | Done automatically by `create_agent` |
| Agent (`create_agent`, ReAct) | Yes | `agent.py` |
| LangGraph (graph, state, nodes) | Yes | Inside `create_agent` |
| Middleware | Yes | `fix_text_tool_calls` (`after_model`) |
| Short-term memory (checkpointer) | Yes | `InMemorySaver` + `thread_id` |
| Streaming | Yes | `stream_mode="updates"` |
| Fake models for testing | Yes | `GenericFakeChatModel` |
| Prompt templates / LCEL chains | No | Not needed: an agent replaces a fixed chain |
| RAG (loaders, embeddings, vector store) | No | Possible extension (see section 11) |
| LangSmith tracing | No | Optional: set `LANGSMITH_TRACING=true` and an API key |

---

## 10. Advantages of This Design

1. **Private and free:** everything runs locally, and no data leaves the laptop except web searches.
2. **Grounded answers:** facts come from real sources with URLs, which gives less hallucination.
3. **Exact maths:** the calculator removes arithmetic errors.
4. **Transparent:** every step of the agent is visible on screen.
5. **Model-independent:** switch LLMs with `--model`.
6. **Extensible:** a new capability is one `@tool` function added to `ALL_TOOLS`.
7. **Robust with small models:** error-returning tools, the text tool-call repair middleware and the report auto-save.
8. **Tested:** 8 automated tests that need no internet.

## 11. Limitations and Future Enhancements

| Limitation | Possible enhancement |
|---|---|
| phi4-mini is small (3.8B), so reports can be short and it may skip tools | Use a bigger model (`qwen2.5:7b`, `llama3.1:8b`) |
| Runs slowly on a CPU-only laptop | GPU, or a smaller or quantised model |
| Memory is lost when the program exits | `SqliteSaver` checkpointer, or a long-term memory Store |
| Cannot answer from the user's own files | **RAG**: load PDFs, split, embed with **`nomic-embed-text`** (already installed), store in **Chroma/FAISS**, and add a retriever tool |
| Terminal only | Web UI with Streamlit or Gradio |
| No monitoring | LangSmith tracing |
| DuckDuckGo may rate-limit | Add Tavily search or a retry middleware |

---

## 12. Summary

ResearchBuddy shows the central idea of LangChain: **an LLM plus tools, a system prompt,
memory and an agent loop gives an autonomous agent**. It uses LangChain's standard
**chat-model interface** (`ChatOllama`), the **`@tool` abstraction**, the v1
**`create_agent`** ReAct agent running on **LangGraph**, **middleware** for reliability,
a **checkpointer** for memory and **streaming** for transparency. The result is a
local agent that researches a topic on the internet and produces a sourced report.
