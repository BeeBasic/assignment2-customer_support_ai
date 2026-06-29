"""
app.py — Interactive CLI Entry Point
=====================================

PURPOSE
-------
The main entry point for running the Customer Support AI system interactively.
Accepts customer queries from the terminal, runs them through the LangGraph
workflow, and displays the detected intent, routing decision, approval flag,
and generated response.

HOW TO RUN
-----------
    cd customer_support_ai
    python app.py

    # Or with PYTHONUTF8 for Windows emoji support:
    $env:PYTHONUTF8=1; python app.py

DEMO MODE
----------
Run all five test queries automatically:
    python app.py --demo

WHAT THIS FILE DOES
--------------------
1. Builds the compiled LangGraph graph (build_graph())
2. Prompts for a Customer ID (used for memory + session tracking)
3. Accepts free-text queries in a loop
4. Calls graph.invoke() with a fresh SupportState each time
5. Displays: intent, approval flag, and the agent's draft response

NOTE ON RESPONSE FIELD
-----------------------
In this module, "agent_draft" is the final output (no Supervisor yet).
When the Supervisor is added in a later module, the display will switch
to state["final_response"].
"""

import os
import sys

# Disable ChromaDB telemetry globally before any Chroma imports occur
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Monkeypatch Posthog capture to disable telemetry printing completely
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

# Set UTF-8 mode for Windows console compatibility
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from state import create_initial_state
from graph import build_graph
from memory.sqlite_memory import init_db, save_turn, clear_history
from langgraph.types import Command

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── ANSI colour codes ─────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SEPARATOR = f"{DIM}{'-' * 60}{RESET}"

# ── Six required demo queries for customer DEMO-001 ─────────────────────────
DEMO_QUERIES = [
    ("DEMO-001", "My name is David. I have a billing issue."),
    ("DEMO-001", "What are your pricing plans available?"),
    ("DEMO-001", "I forgot my password."),
    ("DEMO-001", "My application crashes whenever I upload a file."),
    ("DEMO-001", "I need a refund."),
    ("DEMO-001", "What was my previous support issue?"),
]


def display_result(state: dict) -> None:
    """
    Print the graph result in a clean, formatted layout.

    Args:
        state: The completed SupportState dict returned by graph.invoke().
    """
    intent   = state.get("intent", "unknown")
    approval = state.get("requires_approval", False)

    response = state.get("final_response") or state.get("agent_draft") or "(no response)"

    # Colour-code the intent label
    intent_colour = {
        "sales":        GREEN,
        "technical":    CYAN,
        "billing":      YELLOW,
        "account":      CYAN,
        "memory_recall": DIM,
    }.get(intent, RESET)

    approval_display = (
        f"{RED}YES — Pending Human Approval{RESET}"
        if approval else
        f"{GREEN}No{RESET}"
    )

    print(f"\n{SEPARATOR}")
    print(f"{BOLD}Detected Intent   :{RESET} {intent_colour}{intent.upper()}{RESET}")
    print(f"{BOLD}Approval Required :{RESET} {approval_display}")
    print(f"{BOLD}Response:{RESET}")
    print(f"{SEPARATOR}")
    # Word-wrap the response at 70 chars for readability
    for line in response.split("\n"):
        print(f"  {line}")
    print(SEPARATOR)


def run_query(graph, customer_id: str, query: str) -> dict:
    """
    Build a state, invoke the graph, and return the resulting state.
    Handles HITL pauses by either prompting the user or auto-approving in demo mode.
    """
    state = create_initial_state(customer_id=customer_id, raw_query=query)
    config = {"configurable": {"thread_id": state["session_id"]}}

    result = graph.invoke(state, config=config)

    # ── Check if paused on Human Approval ──────────────────────────────────
    state_snapshot = graph.get_state(config)
    if state_snapshot.next:
        is_demo = "--demo" in sys.argv
        if is_demo:
            print(f"\n{YELLOW}[HITL Review - Auto-Approving for Demo]{RESET}")
            print(f"  Draft: {result.get('agent_draft')}")
            resume_data = {"decision": "approved", "feedback": ""}
        else:
            print(f"\n{YELLOW}┌────────────────────────────────────────────────────────┐")
            print(f"│               HUMAN-IN-THE-LOOP REVIEW REQUIRED         │")
            print(f"└────────────────────────────────────────────────────────┘{RESET}")
            print(f"{BOLD}Customer ID:{RESET} {customer_id}")
            print(f"{BOLD}Agent Draft:{RESET}\n  {result.get('agent_draft')}\n")
            print("Actions:")
            print("  1. Approve (Send to Supervisor)")
            print("  2. Reject Request")
            print("  3. Edit Response Draft")
            choice = input("Select action (1-3): ").strip()
            if choice == "2":
                feedback = input("Reason for rejection (optional): ").strip()
                resume_data = {"decision": "rejected", "feedback": feedback}
            elif choice == "3":
                edited = input("Enter edited response draft:\n> ").strip()
                resume_data = {"decision": "edited", "feedback": edited}
            else:
                resume_data = {"decision": "approved", "feedback": ""}

        # Resume execution
        result = graph.invoke(Command(resume=resume_data), config=config)

    # ── Persist this turn to SQLite memory ─────────────────────────────────
    response_text = result.get("final_response") or result.get("agent_draft") or ""
    session_id    = state["session_id"]

    save_turn(customer_id, session_id, "user",      query)
    save_turn(customer_id, session_id, "assistant", response_text)

    return result


def run_demo(graph) -> None:
    """
    Run all six required demo queries and display results.

    Args:
        graph: The compiled LangGraph graph.
    """
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  DEMO MODE — Running 6 required test queries for DEMO-001{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # Reset SQLite memory for DEMO-001 for clean demonstration
    clear_history("DEMO-001")
    print(f"{DIM}SQLite memory cleared for customer DEMO-001.{RESET}")

    total_queries = len(DEMO_QUERIES)
    for idx, (customer_id, query) in enumerate(DEMO_QUERIES, 1):
        print(f"\n{BOLD}[Query {idx}/{total_queries}]{RESET} {query}")
        result = run_query(graph, customer_id, query)
        display_result(result)

    print(f"\n{GREEN}Demo complete. All {total_queries} queries processed.{RESET}\n")


def run_interactive(graph) -> None:
    """
    Run the interactive CLI loop.

    Args:
        graph: The compiled LangGraph graph.
    """
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  TechCorp AI Customer Support System{RESET}")
    print(f"{DIM}  Powered by LangGraph + qwen2.5:7b (Ollama){RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{DIM}  Commands: 'quit' or 'exit' to stop | '--demo' to run demo{RESET}")

    customer_id = input(f"\n{BOLD}Customer ID{RESET} (press Enter for 'GUEST'): ").strip()
    if not customer_id:
        customer_id = "GUEST"

    print(f"\n{GREEN}Logged in as: {customer_id}{RESET}")

    while True:
        print()
        try:
            query = input(f"{BOLD}Enter query:{RESET}\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Interrupted. Goodbye!{RESET}")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        if query.lower() == "--demo":
            run_demo(graph)
            continue

        try:
            result = run_query(graph, customer_id, query)
            display_result(result)
        except Exception as e:
            print(f"\n{RED}Error processing query: {e}{RESET}")
            print(f"{DIM}Make sure Ollama is running: ollama serve{RESET}")


def main() -> None:
    """
    Main entry point. Builds the graph and runs in demo or interactive mode.
    """
    print(f"{DIM}Initializing database...{RESET}")
    init_db()  # Create conversations table if not exists

    print(f"{DIM}Building graph...{RESET}")
    graph = build_graph()

    if "--demo" in sys.argv:
        run_demo(graph)
    else:
        run_interactive(graph)


if __name__ == "__main__":
    main()
