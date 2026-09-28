"""
TOOLS = the "hands" of the agent.

A LangChain tool is a normal Python function with the @tool decorator.
LangChain sends three things about it to the LLM:
  1. the function NAME        -> what the LLM can call
  2. the DOCSTRING            -> tells the LLM WHEN to use it
  3. the ARGUMENTS/type hints -> a schema the LLM must fill in

The LLM never runs this code. It only *asks* for a call, e.g.
    web_search(query="solar energy India 2025")
LangChain runs the function and gives the result back to the LLM.
"""

import ast
import json
import math
import operator
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain.tools import tool

REPORTS_DIR = Path("reports")
USER_AGENT = "Mozilla/5.0 (LangChain Research Agent - academic project)"


def _fetch(url: str, timeout: int = 15) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- 1. web search
@tool
def web_search(query: str) -> str:
    """Search the internet (DuckDuckGo) for up-to-date information.

    Returns the top results with title, URL and a short snippet.
    Use this first when researching any topic.
    """
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as exc:
        return f"Error: web search failed ({exc})."
    if not results:
        return f"No results for '{query}'."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


# ---------------------------------------------------------------- 2. read a web page
@tool
def read_webpage(url: str) -> str:
    """Open a web page and return its main text (first ~3000 characters).

    Use this after web_search to read the most useful result in detail.
    """
    try:
        html = _fetch(url)
    except Exception as exc:
        return f"Error: could not open {url} ({exc})."
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    if not text:
        return f"The page {url} has no readable text."
    return text[:3000]


# ---------------------------------------------------------------- 3. wikipedia
@tool
def search_wikipedia(topic: str) -> str:
    """Get a reliable encyclopedia summary of a topic from Wikipedia.

    Good for definitions, background, history, people and places.
    """
    try:
        search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}
        )
        hits = json.loads(_fetch(search_url))["query"]["search"]
        if not hits:
            return f"No Wikipedia article found for '{topic}'."
        title = hits[0]["title"]
        page = urllib.parse.quote(title.replace(" ", "_"))
        summary = json.loads(_fetch(f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}"))
        return f"{title} (https://en.wikipedia.org/wiki/{page})\n{summary.get('extract', '')}"
    except Exception as exc:
        return f"Error: Wikipedia lookup failed ({exc})."


# ---------------------------------------------------------------- 4. calculator
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "abs": abs, "round": round}


def _evaluate(node):
    """Safely evaluate a math expression tree (never use eval() on LLM output!)."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _FUNCS:
        return _FUNCS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression exactly, e.g. "(1450 - 1200) / 1200 * 100".

    Always use this for percentages, growth rates, averages or any arithmetic.
    Supports + - * / % ** ( ) and sqrt, log, log10, abs, round.
    """
    try:
        cleaned = expression.replace("^", "**").replace(",", "")
        return f"{expression} = {_evaluate(ast.parse(cleaned, mode='eval'))}"
    except Exception as exc:
        return f"Error: cannot calculate '{expression}' ({exc})."


# ---------------------------------------------------------------- 5. today's date
@tool
def get_today_date() -> str:
    """Return today's date. Use it for questions about 'latest', 'this year', ages, etc."""
    return datetime.now().strftime("%A, %d %B %Y")


# ---------------------------------------------------------------- 6. save the report
@tool
def save_report(title: str, markdown_content: str) -> str:
    """Save a finished research report as a Markdown (.md) file in the 'reports' folder.

    Call this ONCE at the end of a research task, with the complete report.
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    path.write_text(markdown_content, encoding="utf-8")
    return f"Report saved to {path}"


ALL_TOOLS = [web_search, read_webpage, search_wikipedia, calculator, get_today_date, save_report]
