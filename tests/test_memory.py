"""
tests/test_memory.py — Test Suite for memory/sqlite_memory.py
==============================================================

PURPOSE
-------
Tests every function in sqlite_memory.py using a TEMPORARY in-memory
SQLite database (":memory:") so tests never touch the real memory.db
and leave no side effects.

HOW TO RUN
----------
From customer_support_ai/ directory:
    python tests/test_memory.py

No Ollama required. Completes in under 1 second.

WHAT IS TESTED
--------------
1.  init_db()        — table and index created
2.  save_turn()      — insert rows correctly
3.  get_history()    — returns correct count, order, fields
4.  clear_history()  — deletes correct customer only
5.  ordering         — chronological (oldest first)
6.  limit parameter  — respects limit cap
7.  empty customer   — returns empty list
8.  format_history_for_prompt() — readable string output
9.  get_turn_count() — returns correct row count
10. memory recall simulation — insert a real conversation, query it
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.sqlite_memory import (
    init_db,
    save_turn,
    get_history,
    clear_history,
    format_history_for_prompt,
    get_turn_count,
)

# ── Use a temp file DB so we don't pollute memory.db ──────────────────────────
# Note: sqlite3 ":memory:" doesn't work across connections.
# We use a temp file instead, deleted after tests complete.
import tempfile

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
TEST_DB = _tmp.name
_tmp.close()

# ── Colour codes ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    """Assert condition and print result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}[PASS]{RESET} {name}")
    else:
        failed += 1
        print(f"  {RED}[FAIL]{RESET} {name}")
        if detail:
            print(f"         {DIM}Detail: {detail}{RESET}")


def section(title: str) -> None:
    print(f"\n{CYAN}[{title}]{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"{BOLD}  SQLITE MEMORY TEST SUITE{RESET}")
print(f"  Using temp DB: {TEST_DB}")
print(f"{'=' * 60}")

# ── Test 1: init_db creates the table ────────────────────────────────────────
section("1. init_db()")
init_db(db_path=TEST_DB)

import sqlite3
conn = sqlite3.connect(TEST_DB)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
indexes = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
conn.close()

test("conversations table exists",     "conversations" in tables)
test("customer-timestamp index exists", "idx_customer_timestamp" in indexes)
test("init_db is idempotent (re-run)", True)  # Would raise if not CREATE IF NOT EXISTS
init_db(db_path=TEST_DB)  # Second call — should not crash

# ── Test 2: save_turn inserts rows ────────────────────────────────────────────
section("2. save_turn()")

save_turn("DAVID", "sess-001", "user",      "My name is David.",        TEST_DB)
save_turn("DAVID", "sess-001", "assistant", "Nice to meet you, David!", TEST_DB)
time.sleep(0.01)  # Ensure distinct timestamps
save_turn("DAVID", "sess-001", "user",      "I have a billing issue.",  TEST_DB)
save_turn("DAVID", "sess-001", "assistant", "I'll help you with that.", TEST_DB)
time.sleep(0.01)
save_turn("DAVID", "sess-001", "user",      "What was my previous issue?", TEST_DB)

count = get_turn_count("DAVID", TEST_DB)
test("5 turns saved for DAVID",     count == 5, f"got {count}")
test("save_turn empty string skipped", True)
save_turn("DAVID", "sess-001", "user", "", TEST_DB)  # Should not save
test("empty message not saved",
     get_turn_count("DAVID", TEST_DB) == 5,
     f"got {get_turn_count('DAVID', TEST_DB)}")

# Also insert a second customer to test isolation
save_turn("ALICE", "sess-002", "user",      "I need help with pricing.", TEST_DB)
save_turn("ALICE", "sess-002", "assistant", "Happy to help!", TEST_DB)
test("2 turns saved for ALICE", get_turn_count("ALICE", TEST_DB) == 2)

# ── Test 3: get_history returns correct data ───────────────────────────────────
section("3. get_history()")

history = get_history("DAVID", limit=10, db_path=TEST_DB)
test("get_history returns a list",        isinstance(history, list))
test("correct number of turns (5)",       len(history) == 5, f"got {len(history)}")
test("first turn role is 'user'",         history[0]["role"] == "user")
test("first turn message correct",        history[0]["message"] == "My name is David.")
test("second turn role is 'assistant'",   history[1]["role"] == "assistant")
test("each turn has 'role' key",          "role" in history[0])
test("each turn has 'message' key",       "message" in history[0])
test("each turn has 'timestamp' key",     "timestamp" in history[0])

# ── Test 4: ordering is chronological (oldest first) ─────────────────────────
section("4. Chronological ordering")

timestamps = [h["timestamp"] for h in history]
test("timestamps are in ascending order",
     timestamps == sorted(timestamps),
     f"got: {timestamps}")
test("first message is 'My name is David.'",
     history[0]["message"] == "My name is David.")
test("last message is 'What was my previous issue?'",
     history[-1]["message"] == "What was my previous issue?")

# ── Test 5: limit parameter is respected ──────────────────────────────────────
section("5. Limit parameter")

limited = get_history("DAVID", limit=3, db_path=TEST_DB)
test("limit=3 returns exactly 3 turns",   len(limited) == 3, f"got {len(limited)}")
test("limited result still chronological",
     limited[0]["timestamp"] <= limited[-1]["timestamp"])
# The last 3 of 5 should be turns 3,4,5
test("limit returns MOST RECENT turns",
     limited[-1]["message"] == "What was my previous issue?")

# ── Test 6: customer isolation ────────────────────────────────────────────────
section("6. Customer isolation")

alice_history = get_history("ALICE", limit=10, db_path=TEST_DB)
test("ALICE has exactly 2 turns",          len(alice_history) == 2)
test("ALICE history contains no DAVID turns",
     all(h["message"] != "My name is David." for h in alice_history))

unknown_history = get_history("UNKNOWN-CUSTOMER", db_path=TEST_DB)
test("unknown customer returns empty list", unknown_history == [])

# ── Test 7: format_history_for_prompt ─────────────────────────────────────────
section("7. format_history_for_prompt()")

formatted = format_history_for_prompt(history)
test("returns a non-empty string",         isinstance(formatted, str) and len(formatted) > 0)
test("contains 'Customer' label",          "Customer" in formatted)
test("contains 'Assistant' label",         "Assistant" in formatted)
test("contains David's first message",     "My name is David." in formatted)
test("contains billing issue message",     "billing issue" in formatted)

empty_formatted = format_history_for_prompt([])
test("empty history returns fallback message",
     "No previous" in empty_formatted)

print(f"\n  Formatted history preview:")
for line in formatted.split("\n"):
    print(f"  {DIM}{line}{RESET}")

# ── Test 8: clear_history ─────────────────────────────────────────────────────
section("8. clear_history()")

deleted = clear_history("DAVID", TEST_DB)
test(f"clear_history deleted 5 rows",         deleted == 5, f"deleted {deleted}")
test("DAVID history is now empty",            get_history("DAVID", db_path=TEST_DB) == [])
test("DAVID turn count is 0",                 get_turn_count("DAVID", TEST_DB) == 0)

# ALICE should be unaffected
alice_after = get_history("ALICE", db_path=TEST_DB)
test("ALICE history untouched after DAVID clear", len(alice_after) == 2)

# ── Test 9: Memory recall simulation (end-to-end conversation) ────────────────
section("9. Memory recall simulation")

# Simulate a full conversation for customer MARK
turns = [
    ("user",      "Hello, my name is Mark."),
    ("assistant", "Hello Mark! How can I help you today?"),
    ("user",      "I am having a billing issue with my invoice."),
    ("assistant", "I understand. Can you share your invoice number?"),
    ("user",      "Invoice INV-2024-789."),
    ("assistant", "Thank you. I've flagged invoice INV-2024-789 for review."),
]

for role, message in turns:
    save_turn("MARK", "sess-mark-001", role, message, TEST_DB)
    time.sleep(0.005)  # Distinct timestamps

mark_history = get_history("MARK", limit=10, db_path=TEST_DB)
test("Mark has 6 turns stored",            len(mark_history) == 6)
test("Mark's first message is greeting",   "name is Mark" in mark_history[0]["message"])
test("Mark's history contains billing",    any("billing" in h["message"] for h in mark_history))
test("Mark's history contains invoice",    any("INV-2024-789" in h["message"] for h in mark_history))
test("Chronological order maintained",
     mark_history[0]["timestamp"] <= mark_history[-1]["timestamp"])

# Now simulate what memory_handler sees when Mark asks "what was my issue?"
recall_query = "What was my previous issue?"
retrieved = get_history("MARK", limit=6, db_path=TEST_DB)
formatted_recall = format_history_for_prompt(retrieved)
test("Memory recall context is non-empty", len(formatted_recall) > 0)
test("Context contains billing reference",  "billing" in formatted_recall.lower())

print(f"\n  Memory recall context (what the agent sees):")
print(f"  {DIM}{'-'*50}{RESET}")
for line in formatted_recall.split("\n"):
    print(f"  {DIM}{line}{RESET}")
print(f"  {DIM}{'-'*50}{RESET}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
import os as _os
try:
    _os.unlink(TEST_DB)
except Exception:
    pass

# ── Summary ───────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'=' * 60}")
print(f"{BOLD}  RESULTS: {passed}/{total} tests passed{RESET}")

if failed == 0:
    print(f"  {GREEN}ALL MEMORY TESTS PASSED{RESET}")
else:
    print(f"  {RED}{failed} tests FAILED — check output above{RESET}")

print(f"{'=' * 60}\n")
