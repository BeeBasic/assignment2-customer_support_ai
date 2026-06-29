"""
tests/test_classifier.py — Test Suite for agents/classifier.py
================================================================

PURPOSE
-------
Runs the five required test queries through the intent classifier and
verifies that intent and requires_approval are correctly set.

REQUIRES
--------
- Ollama running locally with qwen2.5:7b pulled
- Run from the customer_support_ai/ directory:
    python tests/test_classifier.py

EXPECTED RESULTS
----------------
Query 1: "What are your pricing plans?"   → sales,        approval=False
Query 2: "I forgot my password."          → account,       approval=False
Query 3: "My application crashes."        → technical,     approval=False
Query 4: "I need a refund."               → billing,       approval=True
Query 5: "What was my previous issue?"    → memory_recall, approval=False
"""

import sys
import os

# Allow import from parent (customer_support_ai/) directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import create_initial_state
from agents.classifier import classify_intent

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

# ── Test definitions ──────────────────────────────────────────────────────────
# Format: (query, expected_intent, expected_approval)
TEST_CASES = [
    ("What are your pricing plans?",
     "sales",        False),
    ("I forgot my password.",
     "account",      False),
    ("My application crashes whenever I upload a file.",
     "technical",    False),
    ("I need a refund for my last payment.",
     "billing",      True),
    ("What was my previous support issue?",
     "memory_recall", False),
]


def run_test(idx: int, query: str, expected_intent: str, expected_approval: bool) -> bool:
    """
    Run a single classifier test and print results.

    Args:
        idx:               Test number (1-based)
        query:             Customer query string
        expected_intent:   The intent label we expect
        expected_approval: Whether we expect requires_approval to be True

    Returns:
        True if both intent and approval match expectations, False otherwise.
    """
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{CYAN}Test {idx}: {query}{RESET}")
    print(f"{'─'*60}")

    # Build a minimal state (classifier only needs raw_query)
    state = create_initial_state(
        customer_id=f"TEST-{idx:03d}",
        raw_query=query,
    )

    # Call the classifier node directly
    result = classify_intent(state)

    actual_intent   = result.get("intent", "MISSING")
    actual_approval = result.get("requires_approval", "MISSING")

    # Evaluate
    intent_ok   = (actual_intent == expected_intent)
    approval_ok = (actual_approval == expected_approval)
    passed      = intent_ok and approval_ok

    # Print result table
    print(f"\n  {'Field':<20} {'Expected':<18} {'Got':<18} {'Status'}")
    print(f"  {'─'*70}")
    _print_row("intent",           expected_intent,   actual_intent,   intent_ok)
    _print_row("requires_approval", str(expected_approval), str(actual_approval), approval_ok)

    overall = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n  Overall: [{overall}]")

    return passed


def _print_row(field: str, expected: str, got: str, ok: bool) -> None:
    status = f"{GREEN}OK{RESET}" if ok else f"{RED}MISMATCH{RESET}"
    print(f"  {field:<20} {expected:<18} {got:<18} {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  CLASSIFIER TEST SUITE")
    print(f"  Model: qwen2.5:7b via Ollama")
    print(f"{'='*60}")

    results = []
    for idx, (query, expected_intent, expected_approval) in enumerate(TEST_CASES, 1):
        passed = run_test(idx, query, expected_intent, expected_approval)
        results.append(passed)

    # Summary
    passed_count = sum(results)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed_count}/{total} tests passed")

    if passed_count == total:
        print(f"  {GREEN}ALL TESTS PASSED{RESET}")
    else:
        failed = [i+1 for i, r in enumerate(results) if not r]
        print(f"  {YELLOW}NOTE: Tests {failed} did not match expected intent.{RESET}")
        print(f"  {YELLOW}LLMs may vary slightly. Check if the routed agent makes sense.{RESET}")
        print(f"  {YELLOW}The keyword fallback ensures routing never crashes.{RESET}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
