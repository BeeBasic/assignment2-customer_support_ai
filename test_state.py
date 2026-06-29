"""
test_state.py — Unit Test for state.py
========================================

PURPOSE
-------
Verifies that the SupportState TypedDict and helper functions work correctly
before we build any other module.

HOW TO RUN
----------
From the customer_support_ai directory:
    python test_state.py

EXPECTED OUTPUT
---------------
All 6 tests should print PASS. No imports will fail.
"""

import sys
import os

# ── Allow import from parent directory ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state import SupportState, create_initial_state, format_state_summary

# ── ANSI color codes for clear pass/fail output ──────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

def test(name: str, condition: bool) -> None:
    """Run a single named assertion and print PASS or FAIL."""
    status = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}")
    if not condition:
        sys.exit(1)  # Stop on first failure for clear debugging

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TEST MODULE: state.py")
print("="*60)

# ── Test 1: create_initial_state returns correct types ───────────────────────
print("\n[1] create_initial_state() — basic structure")
state = create_initial_state(
    customer_id="CUST-TEST-001",
    raw_query="What are your pricing plans?"
)
test("customer_id is set correctly",         state["customer_id"] == "CUST-TEST-001")
test("raw_query is set correctly",           state["raw_query"] == "What are your pricing plans?")
test("session_id is auto-generated",         state["session_id"].startswith("CUST-TEST-001-"))
test("intent starts as None",                state["intent"] is None)
test("requires_approval starts as False",    state["requires_approval"] is False)
test("conversation_history starts as None",  state["conversation_history"] is None)
test("retrieved_docs starts as None",        state["retrieved_docs"] is None)
test("agent_draft starts as None",           state["agent_draft"] is None)
test("approval_status starts as None",       state["approval_status"] is None)
test("human_feedback starts as None",        state["human_feedback"] is None)
test("final_response starts as None",        state["final_response"] is None)
test("metadata has timestamp key",           "timestamp" in state["metadata"])

# ── Test 2: explicit session_id is preserved ─────────────────────────────────
print("\n[2] create_initial_state() — explicit session_id")
state2 = create_initial_state(
    customer_id="CUST-002",
    raw_query="I need a refund.",
    session_id="sess-custom-abc"
)
test("explicit session_id is preserved",     state2["session_id"] == "sess-custom-abc")

# ── Test 3: State can be updated like a dict (simulates what nodes do) ────────
print("\n[3] Simulating node updates (dict mutation)")
state["intent"] = "sales"
state["requires_approval"] = False
test("intent can be updated",                state["intent"] == "sales")
test("requires_approval can be toggled",     state["requires_approval"] is False)

# ── Test 4: HITL fields can be set ───────────────────────────────────────────
print("\n[4] HITL field updates")
state["requires_approval"] = True
state["approval_status"] = "pending"
state["human_feedback"] = "Please process this refund."
test("approval_status = pending",            state["approval_status"] == "pending")
state["approval_status"] = "approved"
test("approval_status transitions to approved", state["approval_status"] == "approved")

# ── Test 5: Conversation history format ──────────────────────────────────────
print("\n[5] Conversation history format")
state["conversation_history"] = [
    {"role": "user",      "content": "I had a billing issue last week."},
    {"role": "assistant", "content": "We resolved it by issuing a credit."},
]
test("history has 2 turns",                  len(state["conversation_history"]) == 2)
test("first turn role is user",              state["conversation_history"][0]["role"] == "user")

# ── Test 6: RAG docs format ───────────────────────────────────────────────────
print("\n[6] RAG document chunks format")
state["retrieved_docs"] = [
    {"content": "Our pricing starts at $9/month.", "source": "pricing_guide.pdf"},
    {"content": "Enterprise plans are available.", "source": "pricing_guide.pdf"},
]
test("retrieved_docs has 2 chunks",          len(state["retrieved_docs"]) == 2)
test("chunk has content and source fields",  "content" in state["retrieved_docs"][0])

# ── Test 7: format_state_summary works without errors ────────────────────────
print("\n[7] format_state_summary()")
summary = format_state_summary(state)
test("summary is a non-empty string",        isinstance(summary, str) and len(summary) > 0)
test("summary contains customer_id",         "CUST-TEST-001" in summary)

# ── All tests passed ──────────────────────────────────────────────────────────
print(f"\n{GREEN}{'='*60}")
print("  ALL TESTS PASSED — state.py is working correctly.")
print(f"{'='*60}{RESET}")
print("\nDebug summary of test state:")
print(format_state_summary(state))
