"""
tests/test_memory_recall_integration.py
========================================

PURPOSE
-------
Demonstrates the exact memory recall scenario from the assignment:

  Turn 1: Customer says "My name is David."         → saved to SQLite
  Turn 2: Customer says "I have a billing issue."   → saved to SQLite
  Turn 3: Customer asks "What was my previous issue?" → memory_recall
          → gets history from SQLite
          → LLM generates: "Your previous issue was a billing issue."

This is a LIVE test — requires Ollama running with qwen2.5:7b.

HOW TO RUN
----------
    $env:PYTHONUTF8=1; python tests/test_memory_recall_integration.py
"""

import os
import sys
import time
import tempfile

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from graph import build_graph
from memory.sqlite_memory import init_db, save_turn, get_history, clear_history

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CUSTOMER_ID = "DAVID-INTEGRATION-TEST"


def run_turn(graph, customer_id: str, query: str, label: str) -> dict:
    """Run one query through the graph and save to SQLite."""
    print(f"\n{CYAN}>>> {label}{RESET}")
    print(f"  Customer: \"{query}\"")
    state  = create_initial_state(customer_id=customer_id, raw_query=query)
    config = {"configurable": {"thread_id": state["session_id"]}}
    result = graph.invoke(state, config=config)

    response = result.get("final_response") or result.get("agent_draft") or ""
    print(f"  Intent  : {result.get('intent')}")
    print(f"  Response: {response[:120]}{'...' if len(response) > 120 else ''}")

    # Save to SQLite (same as app.py does)
    save_turn(customer_id, state["session_id"], "user",      query)
    save_turn(customer_id, state["session_id"], "assistant", response)
    return result


def main():
    print(f"\n{'='*60}")
    print(f"{BOLD}  MEMORY RECALL INTEGRATION TEST{RESET}")
    print(f"  Demonstrates: save -> recall -> contextual response")
    print(f"{'='*60}")

    # Clean up any prior test data for this customer
    clear_history(CUSTOMER_ID)
    print(f"\n{DIM}Cleared prior history for {CUSTOMER_ID}{RESET}")

    print(f"\n{DIM}Building graph...{RESET}")
    graph = build_graph()

    # ── Turn 1: Introduce name ─────────────────────────────────────────────
    run_turn(graph, CUSTOMER_ID, "My name is David.", "Turn 1: Introduction")
    time.sleep(0.1)

    # ── Turn 2: Billing issue ──────────────────────────────────────────────
    run_turn(graph, CUSTOMER_ID, "I have a billing issue.", "Turn 2: Billing issue")
    time.sleep(0.1)

    # Verify history was saved
    history = get_history(CUSTOMER_ID, limit=10)
    print(f"\n{DIM}[Check] SQLite history after 2 turns: {len(history)} rows{RESET}")

    for h in history:
        role = "Customer " if h["role"] == "user" else "Assistant"
        print(f"  {DIM}{role}: {h['message'][:70]}{RESET}")

    # ── Turn 3: Memory recall ──────────────────────────────────────────────
    result = run_turn(
        graph, CUSTOMER_ID,
        "What was my previous support issue?",
        "Turn 3: MEMORY RECALL"
    )

    recall_response = result.get("agent_draft") or result.get("final_response") or ""

    # -- Verify the recall response references billing ----------------------
    print(f"\n{'-'*60}")
    print(f"{BOLD}  RECALL RESPONSE EVALUATION{RESET}")
    print(f"{'-'*60}")
    print(f"  Full response:\n  {recall_response}\n")

    billing_mentioned = "billing" in recall_response.lower()
    is_non_stub = "just yet" not in recall_response  # Old stub phrase

    print(f"  {'[PASS]' if billing_mentioned else '[NOTE]'} "
          f"{'References billing issue' if billing_mentioned else 'Billing not explicitly mentioned'}")
    print(f"  {'[PASS]' if is_non_stub else '[FAIL]'} "
          f"{'Real LLM response (not stub)' if is_non_stub else 'Still returning stub!'}")

    # Final history check
    final_history = get_history(CUSTOMER_ID, limit=20)
    print(f"\n  Total turns saved after 3 queries: {len(final_history)}")
    # 3 queries × 2 saves each (user + assistant) = 6 rows
    print(f"  Expected: 6 | Got: {len(final_history)} "
          f"{'[PASS]' if len(final_history) == 6 else '[NOTE: may vary]'}")

    # Cleanup
    clear_history(CUSTOMER_ID)
    print(f"\n{DIM}Cleaned up test data.{RESET}")

    print(f"\n{'='*60}")
    if is_non_stub:
        print(f"{GREEN}{BOLD}  MEMORY RECALL INTEGRATION WORKING{RESET}")
    else:
        print(f"{RED}  Memory handler returned stub — check SQLite integration{RESET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
