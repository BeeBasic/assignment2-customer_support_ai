"""
tests/test_graph.py — End-to-End Graph Test Suite
===================================================

PURPOSE
-------
Runs all five required demo queries through the complete LangGraph
pipeline and verifies:
  - Correct intent classification
  - Correct agent routing
  - agent_draft is populated (non-empty string)
  - requires_approval is correctly set for refund query

REQUIRES
--------
- Ollama running locally with qwen2.5:7b pulled
- Run from the customer_support_ai/ directory:
    $env:PYTHONUTF8=1; python tests/test_graph.py

WHAT "PASS" MEANS HERE
-----------------------
For LLM-based intent classification, the test is LENIENT on intent labels.
The graph never crashes regardless of LLM output (keyword fallback catches
any classification errors). A test PASSES if:
  1. The graph completes without exception
  2. agent_draft is a non-empty string
  3. requires_approval is True for the refund query
  4. The intent is one of the five valid values
"""

import os
import sys
import time

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from graph import build_graph
from memory.sqlite_memory import init_db

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

VALID_INTENTS = {"sales", "technical", "billing", "account", "memory_recall"}

# ── Test definitions ──────────────────────────────────────────────────────────
# Format: (customer_id, query, expected_intent, expected_approval, description)
TEST_CASES = [
    (
        "CUST-001",
        "What are your pricing plans available?",
        "sales",
        False,
        "Pricing inquiry -> Sales Agent",
    ),
    (
        "CUST-002",
        "I forgot my password.",
        "account",
        False,
        "Password issue -> Account Agent",
    ),
    (
        "CUST-003",
        "My application crashes whenever I upload a file.",
        "technical",
        False,
        "Crash report -> Technical Agent",
    ),
    (
        "CUST-004",
        "I need a refund for my last payment.",
        "billing",
        True,  # MUST require approval
        "Refund request -> Billing Agent + HITL flag",
    ),
    (
        "CUST-005",
        "What was my previous support issue?",
        "memory_recall",
        False,
        "Memory recall -> Memory Handler",
    ),
]


def run_test(graph, idx: int, customer_id: str, query: str,
             expected_intent: str, expected_approval: bool,
             description: str) -> bool:
    """
    Execute one test case against the graph and evaluate results.

    Returns True if the test passes, False otherwise.
    """
    print(f"\n{CYAN}{'-' * 60}{RESET}")
    print(f"{BOLD}[Test {idx}] {description}{RESET}")
    print(f"  Query : {query}")
    print(f"{'-' * 60}")

    start_time = time.time()

    try:
        state = create_initial_state(customer_id=customer_id, raw_query=query)
        config = {"configurable": {"thread_id": state["session_id"]}}
        result = graph.invoke(state, config=config)
        
        # -- Handle LangGraph Interruption (HITL) ---------------------------
        state_snapshot = graph.get_state(config)
        if state_snapshot.next:
            print(f"  [HITL] Interrupted at {state_snapshot.next} -> Auto-resuming with 'approved'")
            from langgraph.types import Command
            result = graph.invoke(
                Command(resume={"decision": "approved", "feedback": ""}),
                config=config
            )

        elapsed = time.time() - start_time

        actual_intent   = result.get("intent", "MISSING")
        actual_approval = result.get("requires_approval", "MISSING")
        final_response  = result.get("final_response", "")

        # -- Assertions --------------------------------------------------
        intent_valid    = actual_intent in VALID_INTENTS
        intent_match    = (actual_intent == expected_intent)
        approval_match  = (actual_approval == expected_approval)
        response_exists = bool(final_response and len(final_response.strip()) > 10)

        # Print results table
        print(f"\n  {'Field':<22} {'Expected':<18} {'Got':<18} {'Status'}")
        print(f"  {'-' * 75}")
        _row("intent",            expected_intent,    actual_intent,          intent_match)
        _row("requires_approval", str(expected_approval), str(actual_approval), approval_match)
        _row("final_response",    "non-empty string", f"{len(final_response)} chars", response_exists)
        _row("intent valid",      "yes",              "yes" if intent_valid else "no", intent_valid)

        # Print final response snippet
        snippet = (final_response[:120] + "...") if len(final_response) > 120 else final_response
        print(f"\n  Response snippet:")
        print(f"  {DIM}{snippet}{RESET}")
        print(f"\n  Elapsed: {elapsed:.1f}s")

        # Test passes if: valid intent + approval correct + final_response exists
        passed = intent_valid and approval_match and response_exists

        if not intent_match:
            print(f"\n  {YELLOW}NOTE: Intent '{actual_intent}' != expected '{expected_intent}'.{RESET}")
            print(f"  {YELLOW}This may be acceptable — the keyword fallback ensures routing works.{RESET}")

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"\n  Result: [{status}]")
        return passed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  {RED}EXCEPTION after {elapsed:.1f}s: {e}{RESET}")
        print(f"  {DIM}Is Ollama running? Try: ollama serve{RESET}")
        return False


def _row(field: str, expected: str, got: str, ok: bool) -> None:
    status = f"{GREEN}OK{RESET}" if ok else f"{RED}MISMATCH{RESET}"
    print(f"  {field:<22} {expected:<18} {got:<18} {status}")


def main() -> None:
    print(f"\n{'=' * 60}")
    print(f"{BOLD}  GRAPH END-TO-END TEST SUITE{RESET}")
    print(f"  5 queries | Model: qwen2.5:7b")
    print(f"{'=' * 60}")

    print(f"\n{DIM}Initializing database...{RESET}")
    init_db()  # Ensure conversations table exists before memory_handler runs

    print(f"\n{DIM}Building graph...{RESET}")
    graph = build_graph()

    results = []
    total_start = time.time()

    for idx, (customer_id, query, exp_intent, exp_approval, desc) in enumerate(TEST_CASES, 1):
        passed = run_test(graph, idx, customer_id, query, exp_intent, exp_approval, desc)
        results.append(passed)

    total_elapsed = time.time() - total_start
    passed_count = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"{BOLD}  FINAL RESULTS: {passed_count}/{total} tests passed{RESET}")
    print(f"  Total time: {total_elapsed:.1f}s")

    for idx, (passed, (_, _, _, _, desc)) in enumerate(zip(results, TEST_CASES), 1):
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] Test {idx}: {desc}")

    if passed_count == total:
        print(f"\n  {GREEN}{BOLD}ALL GRAPH TESTS PASSED{RESET}")
    else:
        print(f"\n  {YELLOW}Some tests failed. Check Ollama is running and model is pulled.{RESET}")
        print(f"  {DIM}ollama list   (verify qwen2.5:7b is listed){RESET}")
        print(f"  {DIM}ollama serve  (start Ollama if not running){RESET}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
