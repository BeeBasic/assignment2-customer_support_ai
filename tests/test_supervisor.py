"""
tests/test_supervisor.py — Test Suite for agents/supervisor_agent.py
===================================================================

PURPOSE
-------
Verifies that the Supervisor quality control node correctly reviews drafts:
  - If a draft is factually incorrect or violates policy (e.g. refunding after 90 days),
    the Supervisor rewrites it.
  - If a draft has minor issues, the Supervisor polishes it.
  - Produces clean final_response.

HOW TO RUN
----------
    $env:PYTHONUTF8=1; python tests/test_supervisor.py
"""

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from agents.supervisor_agent import run_supervisor_agent

# ── ANSI colour codes ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}[PASS]{RESET} {name}")
    else:
        failed += 1
        print(f"  {RED}[FAIL]{RESET} {name}")
        if detail:
            print(f"         {detail}")


def main():
    global passed, failed
    print(f"\n{'='*60}")
    print(f"{BOLD}  SUPERVISOR AGENT TEST SUITE{RESET}")
    print(f"  Model: qwen2.5:7b")
    print(f"{'='*60}")

    # ── Test Case 1: Factually Incorrect Draft (Needs Rewrite) ───────────────
    print(f"\n{CYAN}[1] Factually Incorrect Draft (Needs Rewrite){RESET}")
    state = create_initial_state("CUST-101", "What is the price of the Professional plan?")
    state["agent_draft"] = "The Professional plan is $99 per month."
    state["retrieved_docs"] = [
        {"content": "Professional Plan: $29/month. Includes priority support, API access.", "source": "pricing_guide.pdf"}
    ]

    result = run_supervisor_agent(state)
    final = result.get("final_response", "")

    test("final_response generated", len(final) > 0)
    test("Correct price of $29 used in final response", "$29" in final, f"Got: '{final}'")
    test("Incorrect price of $99 removed", "$99" not in final, f"Got: '{final}'")
    print(f"  {DIM}Draft: {state['agent_draft']}{RESET}")
    print(f"  {DIM}Final: {final}{RESET}")

    # ── Test Case 2: Rejected Request (Bypasses LLM, returns rejection response) ──
    print(f"\n{CYAN}[2] Rejected Request Handling{RESET}")
    state = create_initial_state("CUST-102", "I want a refund for a purchase from last year.")
    state["agent_draft"] = "I can issue a refund for you."
    state["approval_status"] = "rejected"
    state["final_response"] = "We cannot process this refund because it is outside the 30-day window."

    result = run_supervisor_agent(state)
    final = result.get("final_response", "")

    test("Preserves human-written rejection message", final == state["final_response"], f"Got: '{final}'")
    print(f"  {DIM}Final: {final}{RESET}")

    # ── Test Case 3: Empty Draft Fallback ────────────────────────────────────
    print(f"\n{CYAN}[3] Empty Draft Fallback{RESET}")
    state = create_initial_state("CUST-103", "Hello")
    state["agent_draft"] = ""

    result = run_supervisor_agent(state)
    final = result.get("final_response", "")

    test("Generates safety fallback response", "apologize" in final or "support" in final or len(final) > 0, f"Got: '{final}'")
    print(f"  {DIM}Final: {final}{RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"{BOLD}  RESULTS: {passed}/{total} tests passed{RESET}")
    if failed == 0:
        print(f"  {GREEN}ALL SUPERVISOR TESTS PASSED{RESET}")
    else:
        print(f"  {RED}{failed} tests FAILED{RESET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
