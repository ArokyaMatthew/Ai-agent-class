"""
Run the Research Agent in your terminal.

    python main.py                      # interactive chat
    python main.py "research electric cars in India and write a report"
    python main.py --model llama3.1:8b  # use a different Ollama model
"""

import argparse
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import DEFAULT_MODEL, build_agent, build_model

console = Console()

EXAMPLES = """[bold]Try asking:[/bold]
  - research the history and future of the James Webb Space Telescope and write a report
  - what is LangChain and who created it?
  - compare solar and wind energy growth in India, with numbers
  - what is 18% GST on 45,999 rupees?

[bold]Commands:[/bold] /new = forget the conversation, /exit = quit"""


def run_turn(agent, question: str, thread_id: str) -> None:
    """Send one question and show every step the agent takes."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    # stream_mode="updates" gives us each step of the ReAct loop as it happens
    with console.status("[cyan]Thinking...", spinner="dots") as status:
        for update in agent.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
            for step in update.values():
                for msg in (step or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for call in msg.tool_calls:
                            args = ", ".join(f"{k}={str(v)[:70]!r}" for k, v in call["args"].items())
                            console.print(f"[yellow]>> Action:[/yellow] [bold]{call['name']}[/bold]({args})")
                            status.update(f"[cyan]Running {call['name']}...")
                    elif isinstance(msg, ToolMessage):
                        preview = str(msg.content).replace("\n", " ")[:150]
                        console.print(f"[dim]   Observation: {preview}...[/dim]")
                        status.update("[cyan]Thinking...")
                    elif isinstance(msg, AIMessage) and msg.text:
                        console.print(Panel(Markdown(msg.text), title="ResearchBuddy",
                                            border_style="green"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain + Ollama research agent")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    agent = build_agent(model=build_model(args.model))
    thread_id = str(uuid.uuid4())

    try:
        if args.question:
            run_turn(agent, args.question, thread_id)
            return

        console.print(Panel(EXAMPLES, title=f"ResearchBuddy  |  LangChain + Ollama ({args.model})",
                            border_style="blue"))
        while True:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
            if question in ("/exit", "/quit", "exit", "quit"):
                break
            if question == "/new":
                thread_id = str(uuid.uuid4())
                console.print("[dim]Started a new conversation.[/dim]")
                continue
            if question:
                run_turn(agent, question, thread_id)
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        if "connect" in str(exc).lower() or "not found" in str(exc).lower():
            console.print(f"[red]Ollama problem:[/red] {exc}\n"
                          f"Is Ollama running, and did you run [bold]ollama pull {args.model}[/bold]?")
        else:
            raise
    console.print("\nBye!")


if __name__ == "__main__":
    main()
