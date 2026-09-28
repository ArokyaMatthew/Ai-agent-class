"""
Offline tests: they run the REAL LangChain agent loop, but with a scripted
"fake" LLM and a fake internet, so no Ollama or network is needed.

    python -m pytest -v
"""

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402
from agent import build_agent  # noqa: E402


class ScriptedLLM(GenericFakeChatModel):
    """A fake chat model that replies with pre-written messages (incl. tool calls)."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, **args):
    return {"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}


@pytest.fixture
def fake_internet(monkeypatch, tmp_path):
    class FakeDDGS:
        def text(self, query, max_results=5):
            return [{"title": "EV sales report", "href": "https://example.com/ev",
                     "body": f"Results about {query}"}]

    pages = {
        "https://example.com/ev": "<html><script>x()</script><p>EV sales grew from 1200 to 1450 units.</p></html>",
    }
    monkeypatch.setattr(tools, "DDGS", FakeDDGS)
    monkeypatch.setattr(tools, "_fetch", lambda url, timeout=15: pages[url])
    monkeypatch.setattr(tools, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path / "reports"


# ------------------------------------------------------------------ tool tests
def test_calculator():
    assert tools.calculator.invoke({"expression": "(1450 - 1200) / 1200 * 100"}).endswith("= 20.833333333333336")
    assert tools.calculator.invoke({"expression": "sqrt(16) + 2^3"}).endswith("= 12.0")
    assert tools.calculator.invoke({"expression": "__import__('os')"}).startswith("Error")


def test_web_search_and_read_page(fake_internet):
    assert "https://example.com/ev" in tools.web_search.invoke({"query": "EV sales"})
    text = tools.read_webpage.invoke({"url": "https://example.com/ev"})
    assert "grew from 1200 to 1450" in text and "x()" not in text


def test_save_report(fake_internet):
    msg = tools.save_report.invoke({"title": "EV Sales!", "markdown_content": "# EV"})
    saved = list(fake_internet.glob("*-ev-sales.md"))
    assert len(saved) == 1 and saved[0].read_text() == "# EV" and "saved" in msg


# ------------------------------------------------------------------ agent tests
def test_agent_runs_full_research_loop(fake_internet):
    """LLM -> web_search -> read_webpage -> calculator -> save_report -> final answer."""
    report = "# EV Sales\n## Summary\nSales grew 20.8%.\n## Sources\n- https://example.com/ev"
    llm = ScriptedLLM(messages=iter([
        AIMessage("", tool_calls=[call("web_search", query="EV sales 2025")]),
        AIMessage("", tool_calls=[call("read_webpage", url="https://example.com/ev")]),
        AIMessage("", tool_calls=[call("calculator", expression="(1450-1200)/1200*100")]),
        AIMessage("", tool_calls=[call("save_report", title="EV Sales", markdown_content=report)]),
        AIMessage("Done! Sales grew about 20.8%. Report saved in the reports folder."),
    ]))
    agent = build_agent(model=llm)

    result = agent.invoke({"messages": [("user", "research EV sales")]},
                          {"configurable": {"thread_id": "t1"}})

    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert len(tool_outputs) == 4
    assert "grew from 1200 to 1450" in tool_outputs[1]
    assert "20.83" in tool_outputs[2]
    assert result["messages"][-1].content.startswith("Done!")
    assert list(fake_internet.glob("*.md"))[0].read_text() == report


def test_agent_remembers_conversation():
    llm = ScriptedLLM(messages=iter([AIMessage("Nice to meet you, Arun!"), AIMessage("Your name is Arun.")]))
    agent = build_agent(model=llm)
    config = {"configurable": {"thread_id": "t2"}}

    agent.invoke({"messages": [("user", "My name is Arun")]}, config)
    result = agent.invoke({"messages": [("user", "What is my name?")]}, config)

    # both turns are stored in the same thread -> memory works
    assert [m.content for m in result["messages"] if m.type == "human"] == ["My name is Arun", "What is my name?"]


# ------------------------------------------------------------------ small-model safety nets
def test_parse_text_tool_calls():
    from agent import parse_text_tool_calls

    calls = parse_text_tool_calls('<|tool_call|>[{"name": "calculator", "arguments": {"expression": "2+2"}}]')
    assert [(c["name"], c["args"]) for c in calls] == [("calculator", {"expression": "2+2"})]
    calls = parse_text_tool_calls('Sure! {"name": "web_search", "parameters": "{\\"query\\": \\"AI\\"}"}')
    assert [(c["name"], c["args"]) for c in calls] == [("web_search", {"query": "AI"})]
    assert parse_text_tool_calls("The answer is [1, 2] and {'a': 1}.") == []
    assert parse_text_tool_calls('{"name": "delete_everything", "arguments": {}}') == []


def test_agent_repairs_tool_call_written_as_text():
    """Model writes the tool call as TEXT -> middleware turns it into a real tool call."""
    llm = ScriptedLLM(messages=iter([
        AIMessage('[{"name": "calculator", "arguments": {"expression": "45999 * 0.18"}}]'),
        AIMessage("18% GST on 45,999 is 8,279.82 rupees."),
    ]))
    result = build_agent(model=llm).invoke({"messages": [("user", "18% GST on 45999?")]},
                                           {"configurable": {"thread_id": "t3"}})

    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert tool_outputs == ["45999 * 0.18 = 8279.82"]
    assert result["messages"][-1].content == "18% GST on 45,999 is 8,279.82 rupees."


def test_report_is_saved_even_if_model_forgets(fake_internet):
    import main

    llm = ScriptedLLM(messages=iter([AIMessage("# Solar Power\n## Summary\nSolar is growing.")]))
    main.run_turn(build_agent(model=llm), "research solar power", "t4")

    saved = list(fake_internet.glob("*-solar-power.md"))
    assert len(saved) == 1 and saved[0].read_text().startswith("# Solar Power")
