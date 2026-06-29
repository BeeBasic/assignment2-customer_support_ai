"""
tests/test_hitl.py — Test Suite for Human-in-the-Loop (HITL) Approval
=====================================================================

PURPOSE
-------
Verifies that the graph pauses on sensitive queries (refunds, cancellations, closures)
and successfully resumes with the appropriate decisions (approved, edited, rejected).
Also verifies that non-sensitive queries bypass HITL entirely.

HOW TO RUN
----------
    $env:PYTHONUTF8=1; python tests/test_hitl.py
"""

import os
import sys
import shutil

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from graph import build_graph
from langgraph.types import Command

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


def section(title: str) -> None:
    print(f"\n{CYAN}[{title}]{RESET}")


def main():
    global passed, failed
    print(f"\n{'='*60}")
    print(f"{BOLD}  HUMAN-IN-THE-LOOP (HITL) TEST SUITE{RESET}")
    print(f"  Model: qwen2.5:7b")
    print(f"{'='*60}")

    print(f"{DIM}Building graph...{RESET}")
    graph = build_graph()

    # ── Test Case 1: Refund query triggers interrupt ──────────────────────────
    section("1. Sensitive Query (Refund) -> Triggers Pause")
    state = create_initial_state("HITL-001", "I need a refund for my purchase.")
    config = {"configurable": {"thread_id": state["session_id"]}}

    # Call graph invoke — should run classifier -> billing_agent -> pause at human_approval
    res = graph.invoke(state, config=config)
    state_snapshot = graph.get_state(config)

    test("requires_approval is True", res.get("requires_approval") is True)
    test("Graph paused (state_snapshot.next is non-empty)", bool(state_snapshot.next))
    test("Paused at human_approval node", state_snapshot.next == ("human_approval",), f"Paused at: {state_snapshot.next}")

    # Resume with APPROVAL
    print(f"\n{DIM}Resuming with 'approved' decision...{RESET}")
    res_approved = graph.invoke(
        Command(resume={"decision": "approved", "feedback": ""}),
        config=config
    )
    state_snapshot_after = graph.get_state(config)

    test("Graph finished execution (snapshot.next is empty)", not state_snapshot_after.next)
    test("final_response generated", len(res_approved.get("final_response", "")) > 0)
    test("approval_status set to approved", res_approved.get("approval_status") == "approved")

    # ── Test Case 2: Resume with EDIT ──────────────────────────────────────────
    section("2. Sensitive Query -> Resume with 'edited' draft")
    state2 = create_initial_state("HITL-002", "I want to cancel my subscription.")
    config2 = {"configurable": {"thread_id": state2["session_id"]}}

    # Invoke up to pause
    res2 = graph.invoke(state2, config=config2)
    state_snapshot2 = graph.get_state(config2)
    test("Graph paused at human_approval node", state_snapshot2.next == ("human_approval",))

    # Resume with EDIT
    human_edit = "I will cancel this for you. Your subscription ends on next bill date."
    print(f"\n{DIM}Resuming with 'edited' draft: '{human_edit}'{RESET}")
    res_edited = graph.invoke(
        Command(resume={"decision": "edited", "feedback": human_edit}),
        config=config2
    )

    test("approval_status set to edited", res_edited.get("approval_status") == "edited")
    test("human_feedback stores edit text", res_edited.get("human_feedback") == human_edit)
    # The supervisor checks the edited draft and polishes/preserves it
    test("final_response incorporates human edit", "ends on next" in res_edited.get("final_response", "").lower(),
         f"Got final: '{res_edited.get('final_response')}'")

    # ── Test Case 3: Resume with REJECT ────────────────────────────────────────
    section("3. Sensitive Query -> Resume with 'rejected'")
    state3 = create_initial_state("HITL-003", "Close my account immediately.")
    config3 = {"configurable": {"thread_id": state3["session_id"]}}

    # Invoke up to pause
    res3 = graph.invoke(state3, config=config3)
    state_snapshot3 = graph.get_state(config3)
    test("Graph paused at human_approval node", state_snapshot3.next == ("human_approval",))

    # Resume with REJECT
    rejection_reason = "Customer has outstanding invoices."
    print(f"\n{DIM}Resuming with 'rejected' decision. Reason: '{rejection_reason}'{RESET}")
    res_rejected = graph.invoke(
        Command(resume={"decision": "rejected", "feedback": rejection_reason}),
        config=config3
    )

    test("approval_status set to rejected", res_rejected.get("approval_status") == "rejected")
    test("final_response contains rejection message and reason", "unable to approve" in res_rejected.get("final_response", "").lower() and "outstanding invoices" in res_rejected.get("final_response", ""),
         f"Got final: '{res_rejected.get('final_response')}'")

    # ── Test Case 4: Normal Query Bypasses HITL ───────────────────────────────
    section("4. Normal Query -> Bypasses human_approval")
    state4 = create_initial_state("HITL-004", "What are the pricing options?")
    config4 = {"configurable": {"thread_id": state4["session_id"]}}

    # Invoke graph - should run sales_agent -> supervisor -> END in one go
    res4 = graph.invoke(state4, config=config4)
    state_snapshot4 = graph.get_state(config4)

    test("requires_approval is False", res4.get("requires_approval") is False)
    test("Graph bypasses human_approval (snapshot.next is empty)", not state_snapshot4.next)
    test("final_response generated directly", len(res4.get("final_response", "")) > 0)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print(f"  {GREEN}ALL HITL TESTS PASSED{RESET}")
    else:
        print(f"  {RED}{failed} tests FAILED{RESET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
