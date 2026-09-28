# ResearchBuddy: a LangChain AI agent that runs on your laptop

ResearchBuddy is an **AI research agent** built with **LangChain** and running on a
**local LLM through Ollama**. It needs no API key, no cloud and no payment.

Give it a topic and it works like a junior researcher:

1. **searches the web** (DuckDuckGo)
2. **opens and reads** the best pages
3. checks **Wikipedia** for background
4. **calculates** numbers exactly (growth %, totals, taxes...)
5. **writes a structured report** with sources and **saves it** as a `.md` file in `reports/`

You can watch every step in the terminal:

```
You: research the growth of electric vehicles in India and write a report
>> Action: web_search(query='electric vehicle sales India 2025')
   Observation: [1] EV sales in India cross ... URL: https://...
>> Action: read_webpage(url='https://...')
   Observation: India sold 1.9 million electric vehicles in ...
>> Action: calculator(expression='(1.9 - 1.5) / 1.5 * 100')
   Observation: (1.9 - 1.5) / 1.5 * 100 = 26.66...
>> Action: save_report(title='Electric Vehicles in India', markdown_content='# Electric ...')
   Observation: Report saved to reports/20260928-101500-electric-vehicles-in-india.md
╭──────────────── ResearchBuddy ────────────────╮
│ I researched ... Report saved to reports/...  │
╰───────────────────────────────────────────────╯
```

**Study notes:**
- [docs/1_LangChain_Framework.md](docs/1_LangChain_Framework.md): LangChain definition, architecture, components and ecosystem (long-answer notes)
- [docs/2_Project_Explanation.md](docs/2_Project_Explanation.md): how this project works, its tools and the LangChain features it uses

---

## 1. Setup (one time)

**Step 1: Install Ollama.** Download it from <https://ollama.com/download> (Windows / Mac / Linux).

**Step 2: Get a model.** The default is **`phi4-mini`** (2.5 GB, supports tool calling).
If `ollama list` already shows it, skip this step. Otherwise:

```bash
ollama pull phi4-mini
```

> `nomic-embed-text` is an *embedding* model. It turns text into numbers for search and
> can't chat or call tools, so it can't be the agent's brain.
>
> Want better reports? Bigger models follow instructions better: `ollama pull qwen2.5:7b`
> (4.7 GB), then run `python main.py --model qwen2.5:7b`. `llama3.1:8b` and `qwen3:8b` also work.

**Step 3: Install the Python packages** (Python 3.10+):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run it

```bash
python main.py
```

Or ask one question directly:

```bash
python main.py "research the James Webb Space Telescope and write a report"
python main.py --model qwen2.5:7b "what is LangChain and who created it?"
```

In chat: `/new` starts a fresh conversation, `/exit` quits.

**Good demo prompts:**
- `research renewable energy in India with numbers and write a report`
- `what are the latest developments in quantum computing? give sources`
- `what is 18% GST on 45,999 rupees?`
- `My name is Priya.` then later `what is my name?` (shows memory)

## 3. Run the tests (no Ollama needed)

```bash
python -m pytest -v
```

The tests run the **real LangChain agent loop** with a scripted fake LLM and a fake
internet. They check that the agent calls the tools in order, passes results back
and saves the report, and that it remembers the conversation.

---

## 4. How it works: LangChain concepts

| LangChain concept | What it means | Where in the code |
|---|---|---|
| **Chat model** | The "brain": an LLM that reads messages and decides what to do | `build_model()` in `agent.py` (`ChatOllama`) |
| **Tools** | Python functions the LLM can ask to run. The `@tool` decorator turns the name, docstring and arguments into a schema the LLM understands | `tools.py` |
| **System prompt** | Instructions that define the agent's role and rules | `SYSTEM_PROMPT` in `agent.py` |
| **Agent** | Loop that connects the model and tools: `create_agent(model, tools, system_prompt)` | `build_agent()` in `agent.py` |
| **Middleware** | Code that runs inside the agent loop. Ours repairs tool calls that small models write as plain text | `fix_text_tool_calls` in `agent.py` |
| **Memory (checkpointer)** | Saves the conversation per `thread_id` so the agent remembers earlier messages | `InMemorySaver()` in `agent.py` |
| **Streaming** | Shows each step (tool call, result, answer) as it happens | `run_turn()` in `main.py` |

### The ReAct loop (Reason → Act → Observe)

```
 your question
      │
      ▼
 ┌──────────┐   "I need a tool"    ┌──────────┐
 │  MODEL   │ ───────────────────▶ │  TOOLS   │  web_search, read_webpage,
 │ (Ollama) │ ◀─────────────────── │ (Python) │  search_wikipedia, calculator,
 └──────────┘    tool result       └──────────┘  get_today_date, save_report
      │
      │ "I have enough information"
      ▼
 final answer / saved report
```

1. The user's message and the system prompt go to the LLM, together with a description of all tools.
2. The LLM either **answers** or returns a **tool call**, e.g. `web_search(query="...")`.
3. LangChain **runs the Python function** and sends the result back to the LLM as a `ToolMessage`.
4. Steps 2-3 repeat until the LLM has enough information and writes the final answer.

The LLM never runs code itself. It only decides which tool to use and with which
arguments. That decision-making is what makes it an **agent** and not a plain chatbot.

### The tools

| Tool | What it does |
|---|---|
| `web_search` | Searches DuckDuckGo, returns top 5 titles, URLs and snippets |
| `read_webpage` | Downloads a page and extracts its readable text |
| `search_wikipedia` | Gets a Wikipedia summary of a topic |
| `calculator` | Safe math (parses the expression, never uses `eval`) |
| `get_today_date` | Today's date, for "latest" / "this year" questions |
| `save_report` | Writes the final Markdown report into `reports/` |

To add your own tool: write a function with `@tool` and a clear docstring in
`tools.py`, then add it to `ALL_TOOLS`. The agent can use it right away.

## 5. Project structure

```
├── agent.py           # model + tools + prompt + memory → the agent
├── tools.py           # the 6 tools
├── main.py            # terminal app (shows each agent step)
├── tests/
│   └── test_agent.py  # offline tests with a fake LLM
├── requirements.txt
└── reports/           # created automatically; your saved reports
```

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama problem: ... connect` | Start Ollama (open the app, or run `ollama serve`) |
| `model "phi4-mini" not found` | Run `ollama pull phi4-mini` |
| `does not support tools` | Update Ollama to the latest version, or use `--model qwen2.5:7b` |
| Very slow answers | Normal on a laptop without a GPU: a full report can take a few minutes |
| Report is short or misses facts | Small model limit. Try `--model qwen2.5:7b`, or ask it to read more pages |
| Agent doesn't use tools | Use a model with tool support (phi4-mini, qwen2.5, llama3.1, qwen3) |
| `web_search` errors | Check your internet connection; DuckDuckGo sometimes rate-limits, so wait a minute |
