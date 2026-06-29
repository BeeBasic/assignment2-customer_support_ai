"""
tests/test_router.py — Test Suite for utils/router.py
=======================================================

PURPOSE
-------
Verifies every routing path in both route_by_intent and approval_router
WITHOUT calling the LLM (pure state dict manipulation).

HOW TO RUN
----------
From customer_support_ai/ directory:
    python tests/test_router.py

EXPECTED OUTPUT
---------------
All tests should PASS. No Ollama required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from utils.router import route_by_intent, approval_router

GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

passed_count = 0
total_count  = 0


def test(name: str, actual: str, expected: str) -> None:
    """Assert actual == expected and print result."""
    global passed_count, total_count
    total_count += 1
    ok = (actual == expected)
    if ok:
        passed_count += 1
    status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}")
    if not ok:
        print(f"         Expected: '{expected}' | Got: '{actual}'")


def make_state(intent: str, requires_approval: bool = False) -> dict:
    """Build a minimal state dict for router testing."""
    state = create_initial_state(customer_id="ROUTER-TEST", raw_query="test query")
    state["intent"] = intent
    state["requires_approval"] = requires_approval
    return state


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ROUTER TEST SUITE")
print(f"  (No Ollama required — pure Python)")
print(f"{'='*60}")

# ── Section 1: route_by_intent — all five intents ─────────────────────────────
print(f"\n{CYAN}[1] route_by_intent — all five valid intents{RESET}")

test("sales → sales_agent",
     route_by_intent(make_state("sales")),
     "sales_agent")

test("technical → technical_agent",
     route_by_intent(make_state("technical")),
     "technical_agent")

test("billing → billing_agent",
     route_by_intent(make_state("billing")),
     "billing_agent")

test("account → account_agent",
     route_by_intent(make_state("account")),
     "account_agent")

test("memory_recall → memory_handler",
     route_by_intent(make_state("memory_recall")),
     "memory_handler")

# ── Section 2: route_by_intent — edge cases ───────────────────────────────────
print(f"\n{CYAN}[2] route_by_intent — edge cases{RESET}")

test("empty string → falls back to account_agent",
     route_by_intent(make_state("")),
     "account_agent")

test("unknown intent → falls back to account_agent",
     route_by_intent(make_state("unknown_xyz")),
     "account_agent")

test("None intent → falls back to account_agent",
     route_by_intent(make_state(None)),
     "account_agent")

test("uppercase SALES → case-insensitive → sales_agent",
     route_by_intent(make_state("SALES")),
     "sales_agent")

# ── Section 3: approval_router — both outcomes ────────────────────────────────
print(f"\n{CYAN}[3] approval_router — HITL vs Supervisor{RESET}")

test("requires_approval=True → human_approval",
     approval_router(make_state("billing", requires_approval=True)),
     "human_approval")

test("requires_approval=False → supervisor",
     approval_router(make_state("sales", requires_approval=False)),
     "supervisor")

test("any intent with approval=True → human_approval",
     approval_router(make_state("account", requires_approval=True)),
     "human_approval")

test("any intent with approval=False → supervisor",
     approval_router(make_state("technical", requires_approval=False)),
     "supervisor")

# ── Section 4: verify five test queries route correctly ───────────────────────
print(f"\n{CYAN}[4] Five demo queries — end-to-end routing check{RESET}")

demo_routes = [
    ("sales",         "sales_agent",      "Query 1: pricing plans"),
    ("account",       "account_agent",    "Query 2: forgot password"),
    ("technical",     "technical_agent",  "Query 3: app crashes"),
    ("billing",       "billing_agent",    "Query 4: refund"),
    ("memory_recall", "memory_handler",   "Query 5: previous issue"),
]

for intent, expected_node, label in demo_routes:
    test(label, route_by_intent(make_state(intent)), expected_node)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {passed_count}/{total_count} tests passed")
if passed_count == total_count:
    print(f"  {GREEN}ALL ROUTER TESTS PASSED{RESET}")
else:
    print(f"  {RED}{total_count - passed_count} tests FAILED — check output above{RESET}")
print(f"{'='*60}\n")
