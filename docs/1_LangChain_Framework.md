# LangChain Framework: Architecture, Components and Tools

*(Written as a long answer for a 10–20 mark question. Headings match the points
an examiner looks for. Diagrams can be drawn directly in the exam.)*

---

## 1. Definition

**LangChain** is an **open-source framework for developing applications powered by
Large Language Models (LLMs)**. It gives developers standard building blocks for
connecting an LLM to **external data sources**, **tools (APIs, functions)**,
**memory** and **other components**, so the LLM becomes part of a larger application
instead of a stand-alone text generator.

- **Created by:** Harrison Chase, released in **October 2022**; now maintained by the company LangChain Inc.
- **Languages:** Python (`langchain`) and JavaScript/TypeScript (`langchainjs`).
- **Current version:** LangChain **1.x** (v1.0 released October 2025). It is centred on
  **agents** and built on top of **LangGraph**.
- **Core idea:** *"chain"* together components (prompt → model → parser → tool → …)
  to build complex LLM workflows.

> **One-line definition for exams:** LangChain is a modular, open-source framework that
> simplifies building LLM applications by providing standard interfaces for models,
> prompts, tools, memory and retrieval, and by composing them into chains and agents.

---

## 2. Need for LangChain (Limitations of a plain LLM)

| Limitation of a plain LLM | How LangChain solves it |
|---|---|
| Knowledge is frozen at training time | **Retrieval (RAG)** and **tools** such as web search bring in fresh data |
| Cannot act (no API calls, no file access) | **Tools + Agents** let the LLM call functions |
| Stateless: forgets earlier messages | **Memory** (checkpointers, message history) |
| Hallucinations on private or domain data | **RAG** grounds answers in your own documents |
| Every provider has a different API | **Standard model interface**: switch OpenAI → Gemini → Ollama with one line |
| Free-text output is hard for programs to use | **Output parsers / structured output** (JSON, Pydantic) |
| Multi-step logic is hard to write | **Chains, LCEL, agents, LangGraph** |
| Hard to debug or evaluate | **LangSmith** tracing and evaluation |

---

## 3. Architecture of LangChain

LangChain is a **layered, modular architecture** split into separate packages.

```
┌──────────────────────────────────────────────────────────────────┐
│                     LangSmith  (Observability)                   │
│         Tracing · Debugging · Evaluation · Monitoring · Deploy   │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                  YOUR LLM APPLICATION                            │
│        Chatbot · RAG Q&A · Agent · Summariser · Extractor        │
└──────────────────────────────────────────────────────────────────┘
┌─────────────────────────────┐  ┌─────────────────────────────────┐
│  langchain  (v1)            │  │  LangGraph  (Orchestration)     │
│  create_agent, middleware,  │◄─┤  Stateful graphs, loops,        │
│  high-level app components  │  │  persistence, human-in-the-loop │
└─────────────────────────────┘  └─────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  Integration packages          langchain-openai, langchain-ollama│
│  (one per provider)            langchain-google-genai, -groq,    │
│  + langchain-community         -chroma, -pinecone ... (1000+)    │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  langchain-core   (Foundation / base abstractions)               │
│  Runnable interface · LCEL · Messages · Prompts · Chat models    │
│  Tools · Output parsers · Documents · Retrievers · Callbacks     │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Packages / layers

| Layer (package) | Role |
|---|---|
| **langchain-core** | Base abstractions and interfaces: `Runnable`, messages, prompt templates, `BaseChatModel`, `BaseTool`, output parsers, `Document`, `BaseRetriever`, embeddings and vector-store interfaces, callbacks. Very few dependencies. Every other package builds on it. |
| **Integration packages** | One small package per provider that implements the core interfaces, e.g. `langchain-openai` (`ChatOpenAI`), `langchain-ollama` (`ChatOllama`), `langchain-google-genai`, `langchain-anthropic`, `langchain-chroma`. |
| **langchain-community** | Community-maintained integrations: document loaders, vector stores and tools that have no dedicated package. |
| **langchain** (v1) | The main package. Provides `create_agent` (the standard agent), **middleware**, `init_chat_model`, and the `@tool` decorator. |
| **langchain-classic** | Legacy v0.x features (old `LLMChain`, `AgentExecutor`, old memory classes, `RetrievalQA`) moved here in v1, kept for backward compatibility. |
| **LangGraph** | Low-level **orchestration framework** for stateful, multi-step, multi-actor applications modelled as **graphs**. LangChain v1 agents run on LangGraph. |
| **LangSmith** | Platform for **tracing, debugging, testing, evaluating and monitoring** LLM apps; also hosts deployment (formerly "LangGraph Platform"). Works with or without LangChain. |
| **LangServe** *(older)* | Deploys chains as REST APIs using FastAPI. Now in maintenance mode; LangSmith deployment is recommended instead. |

### 3.2 Design principles
1. **Modularity:** every component is independent and swappable.
2. **Standard interfaces:** every model, retriever and tool follows one API.
3. **Composability:** components combine using LCEL or graphs.
4. **Provider-agnostic:** no lock-in to one LLM vendor.
5. **Production-oriented:** streaming, async, batching, retries and tracing are built in.

---

## 4. Core Components (Modules) of LangChain

### 4.1 Models (Model I/O)
LangChain gives a **uniform interface** to two kinds of language model:
- **LLMs (text completion):** input is a string, output is a string. This is the older style.
- **Chat Models:** input is a list of **messages**, output is a message. This is the modern standard, e.g. `ChatOpenAI`, `ChatOllama`, `ChatGoogleGenerativeAI`.
- `init_chat_model("provider:model")` creates any provider's chat model from one string.
- Common parameters include `temperature` (randomness), `max_tokens`, `timeout` and `max_retries`.
- **Key capabilities:** tool calling (`bind_tools`), structured output (`with_structured_output`), streaming, multimodal input.

### 4.2 Messages
A conversation is a list of typed messages:

| Message | Sent by | Purpose |
|---|---|---|
| `SystemMessage` | developer | Instructions or persona for the model |
| `HumanMessage` | user | User input |
| `AIMessage` | model | Model reply; can contain `tool_calls` |
| `ToolMessage` | tool | Result of a tool call, linked by `tool_call_id` |

### 4.3 Prompts (Prompt Templates)
Reusable templates with variables, instead of hard-coded strings.
- `PromptTemplate`: for plain-text prompts. Example: `"Summarise {text} in {n} lines"`.
- `ChatPromptTemplate`: a list of message templates (system, human, and so on).
- `MessagesPlaceholder`: inserts a whole chat history into a prompt.
- **Few-shot prompt templates:** add worked examples to guide the model.
- The **LangChain Hub** in LangSmith stores and versions shared prompts.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    ("human", "Explain {topic} in simple words."),
])
```

### 4.4 Output Parsers and Structured Output
These convert raw LLM text into data a program can use.
- `StrOutputParser` returns plain text; `JsonOutputParser` returns a dict; `PydanticOutputParser` returns a typed object.
- Modern approach: `model.with_structured_output(MyPydanticClass)`. This uses the provider's native JSON or tool mode.

### 4.5 Runnables and LCEL (LangChain Expression Language)
- **Runnable** is the universal interface. Every component (prompt, model, parser, retriever, tool, chain) is a Runnable and supports the same methods:
  `invoke()` (one input), `batch()` (many inputs), `stream()` (token by token), and async versions (`ainvoke`, `astream`, …).
- **LCEL** composes Runnables with the **pipe operator `|`**, just like Unix pipes:

```python
chain = prompt | model | StrOutputParser()
chain.invoke({"topic": "photosynthesis"})
```

| LCEL primitive | Purpose |
|---|---|
| `RunnableSequence` | Runs steps one after another (what `a | b | c` creates) |
| `RunnableParallel` | Runs several branches at the same time on the same input |
| `RunnablePassthrough` | Passes the input through unchanged, or adds keys to it |
| `RunnableLambda` | Wraps any Python function as a Runnable |
| `RunnableBranch` | Chooses a branch based on a condition (if/else routing) |
| `.with_fallbacks()`, `.with_retry()` | Built-in reliability |

Benefits: streaming, async, parallelism and tracing come automatically.

### 4.6 Chains
A **chain** is a **fixed, pre-defined sequence of steps**, for example prompt → LLM → parser.
- Old style: `LLMChain`, `SequentialChain`, `RetrievalQA` (now in `langchain-classic`).
- Modern style: chains are written with LCEL.
- **Chain vs Agent:** in a chain the **developer** fixes the order of steps. In an agent the **LLM decides** the next step at run time.

### 4.7 Tools
A **tool** is a function the LLM can ask to run: web search, a calculator, a database query, an API call.
- It is created with the `@tool` decorator. The **name**, **docstring** (description) and **argument schema** (from type hints) are sent to the LLM.
- `model.bind_tools([tools])` tells the model which tools exist.
- **Tool calling flow:** the model returns `AIMessage.tool_calls` (tool name and arguments), the application runs the function, the result goes back as a `ToolMessage`, and the model continues.
- **Toolkits** are groups of related tools, e.g. an SQL toolkit or a Gmail toolkit.
- Ready-made tools include web search (Tavily, DuckDuckGo), Wikipedia, Python REPL, SQL database, file system and many APIs.

### 4.8 Agents
An **agent** uses an **LLM as a reasoning engine** to decide **which actions to take, in what order**, based on the results it observes, until the task is done.

**ReAct pattern (Reason + Act):** the pattern agents are built on.
```
Thought → Action (tool call) → Observation (tool result) → Thought → … → Final Answer
```

**Agent loop in LangChain v1:**
```
        ┌─────────────┐
 Input →│    MODEL    │── no tool call ──→ Final answer
        └─────┬───────┘
    tool_calls│   ▲ ToolMessage (observation)
              ▼   │
        ┌─────────────┐
        │    TOOLS    │
        └─────────────┘
```

- **`create_agent(model, tools, system_prompt, middleware, checkpointer)`** is the standard v1 API. It returns a LangGraph graph with a `model` node and a `tools` node.
- The older `AgentExecutor` and `initialize_agent` are legacy and now in `langchain-classic`.
- **Middleware (v1):** hooks that run inside the agent loop to customise it:
  `before_agent`, `before_model`, `wrap_model_call`, `after_model`, `wrap_tool_call`, `after_agent`.
  Built-in examples:
  - **Summarization:** compresses long history.
  - **Human-in-the-Loop:** asks for approval before a tool runs.
  - **PII:** redacts personal data.
  - **Model call limit** and **tool call limit.**
  - **Model fallback** and **tool retry.**
  - **To-do list:** lets the agent plan.
- **Types of agents:**
  - Tool-calling (ReAct) agents.
  - Structured-output agents (`response_format`).
  - Multi-agent systems: a supervisor with sub-agents, or agent hand-offs.

### 4.9 Memory
Memory lets an application remember past interactions.
- **Short-term memory (thread-level):** the conversation history of one session. In v1 it is stored by a LangGraph **checkpointer** (`InMemorySaver`, `SqliteSaver`, `PostgresSaver`) and keyed by a `thread_id`.
- **Long-term memory (cross-session):** facts kept across conversations, held in a **Store** (`InMemoryStore`, database-backed stores).
- **Legacy memory classes:** `ConversationBufferMemory` (full history), `ConversationBufferWindowMemory` (last k turns), `ConversationSummaryMemory` (LLM summary), `VectorStoreRetrieverMemory`.
- A context window is limited, so long histories are **trimmed** or **summarised**.

### 4.10 Retrieval and RAG (Retrieval-Augmented Generation)
RAG gives an LLM **external or private knowledge** at query time, which reduces hallucination.

**Components:**

| Component | Role | Examples |
|---|---|---|
| **Document Loaders** | Load data into `Document` objects (text plus metadata) | `PyPDFLoader`, `WebBaseLoader`, `CSVLoader`, `TextLoader`, Notion, YouTube |
| **Text Splitters** | Split long documents into chunks that fit the context window | `RecursiveCharacterTextSplitter` (with chunk_size and chunk_overlap), token-based and Markdown splitters |
| **Embedding Models** | Convert text into numeric vectors that capture meaning | `OpenAIEmbeddings`, `OllamaEmbeddings` (e.g. `nomic-embed-text`), HuggingFace |
| **Vector Stores** | Store vectors and run similarity search | FAISS, Chroma, Pinecone, Qdrant, Weaviate, pgvector, Milvus |
| **Retrievers** | Return the relevant documents for a query | `vectorstore.as_retriever()`, MultiQuery, Contextual Compression, BM25, Ensemble (hybrid), Parent-Document |

**RAG pipeline:**
```
INDEXING (offline):  Load → Split → Embed → Store in Vector DB

RETRIEVAL + GENERATION (per query):
  Question → Embed → Similarity search (top-k chunks) →
  Prompt = question + retrieved context → LLM → Grounded answer
```

A RAG chain can be a fixed LCEL chain, or **agentic RAG**, where the retriever is given to the agent as a tool.

### 4.11 Callbacks, Streaming and Tracing
- **Callbacks** hook into events (LLM start/end, tool start/end, errors) for logging, monitoring and progress bars.
- **Streaming** sends tokens or agent steps as they are produced (`stream()`, with `stream_mode="updates"` or `"messages"`).
- **Tracing:** setting `LANGSMITH_TRACING=true` sends every run to LangSmith.

---

## 5. LangGraph (Orchestration Layer)

**LangGraph** is a library, from the same company, for building **stateful, cyclic,
multi-step LLM workflows as graphs**. LangChain v1 agents are compiled LangGraph graphs.

**Concepts:**

| Concept | Meaning |
|---|---|
| **State** | A shared data object (e.g. a `messages` list) passed between nodes |
| **Nodes** | Python functions that read state and return updates (an LLM call, a tool run) |
| **Edges** | Fixed transitions between nodes |
| **Conditional edges** | Routing decided at run time (e.g. "tool call? go to tools, otherwise END") |
| **Reducers** | Rules for merging updates into state (e.g. `add_messages` appends messages) |
| **START / END** | Entry and exit points |
| **Checkpointer** | Saves state after every step: memory, fault tolerance, "time travel" |
| **Interrupts** | Pause for **human-in-the-loop** approval, then resume |

**Why graphs?** Chains are linear (a DAG, with no loops). Agents need **loops** (think → act → observe → think again), **branching** and **persistence**, and graphs support all three.

```python
graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", run_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "model")
app = graph.compile(checkpointer=InMemorySaver())
```

Multi-agent patterns in LangGraph include **supervisor**, **hierarchical teams**, **swarm / hand-off** and **plan-and-execute**.

---

## 6. LangSmith (Observability and Evaluation Platform)

| Feature | Use |
|---|---|
| **Tracing** | Shows every step of a run (prompts, model outputs, tool calls, latency, token cost) |
| **Debugging** | Find the step where an agent went wrong |
| **Datasets and Evaluation** | Test an app on example inputs, scored by LLM-as-judge or custom evaluators |
| **Monitoring** | Dashboards for cost, latency and errors in production |
| **Prompt Hub** | Version and share prompts |
| **Deployment** | Host LangGraph agents as scalable APIs |

---

## 7. Frameworks and Tools in the LangChain Ecosystem (Summary Table)

| Category | Tools / Frameworks |
|---|---|
| **Core framework** | langchain-core, langchain, langchain-classic, langchain-community |
| **Orchestration** | LangGraph |
| **Observability and evaluation** | LangSmith |
| **Deployment** | LangSmith Deployment (formerly LangGraph Platform), LangServe (legacy) |
| **LLM providers** | OpenAI, Anthropic, Google Gemini, Mistral, Groq, Cohere, AWS Bedrock, Azure OpenAI, HuggingFace, **Ollama (local)** |
| **Embeddings** | OpenAI, HuggingFace sentence-transformers, Ollama (`nomic-embed-text`), Cohere, Google |
| **Vector databases** | FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus, pgvector, Elasticsearch |
| **Document loaders** | PDF, Word, CSV, HTML/Web, Notion, Google Drive, YouTube transcripts, GitHub |
| **Tools** | DuckDuckGo, Tavily, Wikipedia, arXiv, Python REPL, SQL database, Requests, Gmail, Slack |
| **UI / apps** | Streamlit, Gradio, Chainlit, FastAPI (commonly paired with LangChain) |
| **Related / alternative frameworks** | LlamaIndex (data and RAG focus), Haystack (search/RAG pipelines), Microsoft Semantic Kernel, AutoGen and CrewAI (multi-agent) |

---

## 8. Working of a LangChain Application (Step by Step)

1. **Select a model** through a standard chat-model interface (`ChatOllama`, `ChatOpenAI` …).
2. **Design the prompt** with a template or a system prompt.
3. **Add external knowledge** (RAG): load, split, embed, store, retrieve.
4. **Add tools** so the model can act.
5. **Compose the logic:** a fixed chain (LCEL) or a dynamic agent (`create_agent` / LangGraph).
6. **Add memory** (checkpointer and `thread_id`) for multi-turn conversation.
7. **Parse the output** into text or structured data.
8. **Trace, evaluate and deploy** with LangSmith.

---

## 9. Applications

- Chatbots and virtual assistants with memory
- **Question answering over documents (RAG):** PDFs, company knowledge bases, legal and medical texts
- **Autonomous agents:** research assistants, coding assistants, customer support
- Text summarisation (documents, meetings, videos)
- Information extraction into structured data (invoices, resumes)
- SQL / data-analysis agents (natural language → SQL)
- Content generation, translation, tutoring systems
- Multi-agent workflows (planner + researcher + writer)

---

## 10. Advantages

1. **Modular and reusable** components.
2. **Model-agnostic:** switch providers without rewriting code; avoids vendor lock-in.
3. **Huge integration ecosystem** (1000+ integrations).
4. **Speeds up development** of RAG and agents with ready-made abstractions.
5. **Built-in production features:** streaming, async, batching, retries, fallbacks.
6. **Observability** with LangSmith.
7. **Supports local, private models** through Ollama.
8. Large community, open source (MIT licence), Python and JS versions.

## 11. Limitations

1. **Abstraction overhead:** layers can hide what is happening, which makes debugging harder.
2. **Fast-changing API:** frequent breaking changes (v0.1 → v0.2 → v0.3 → v1.0); tutorials go out of date.
3. **Learning curve:** many concepts (Runnables, LCEL, LangGraph, middleware).
4. **Performance overhead** compared with calling a model API directly.
5. **Quality still depends on the LLM:** LangChain cannot remove hallucination; weak models make weak agents.
6. **Dependency bloat** in large projects.

---

## 12. Conclusion

LangChain turns an LLM from an isolated text generator into the **reasoning core of an
application**. It does this through standardised **models, prompts, tools, memory and
retrieval**, composed with **LCEL chains** or **agents** that run on **LangGraph**, and
monitored with **LangSmith**. Its modular, provider-independent architecture made it
one of the most widely used frameworks for building RAG systems and AI agents.

---

### Exam tips
- **Always draw:** (1) the layered architecture (§3), (2) the agent / ReAct loop (§4.8), (3) the RAG pipeline (§4.10).
- **For 10 marks:** definition, need, architecture diagram, 6 components (models, prompts, chains, tools, agents, memory), RAG, advantages.
- **For 20 marks:** add LCEL, LangGraph, LangSmith, the ecosystem table, applications, limitations and a small code snippet.
