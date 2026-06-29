"""
tests/test_rag.py — Test Suite for RAG Module
==============================================

Verifies retrieval for all four query types: pricing, password, crash, refund.
Requires Ollama running (nomic-embed-text for embeddings).

Run from customer_support_ai/:
    $env:PYTHONUTF8=1; python tests/test_rag.py
"""

import os
import sys
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve, format_docs_for_prompt
from rag.vectorstore import is_vectorstore_built, build_vectorstore

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
BOLD  = "\033[1m";  DIM = "\033[2m";  RESET = "\033[0m"

passed = 0; failed = 0

def test(name, cond, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"  {GREEN}[PASS]{RESET} {name}")
    else:
        failed += 1; print(f"  {RED}[FAIL]{RESET} {name}")
        if detail: print(f"         {DIM}{detail}{RESET}")

def section(title): print(f"\n{CYAN}[{title}]{RESET}")

print(f"\n{'='*60}\n{BOLD}  RAG TEST SUITE{RESET}\n  nomic-embed-text via Ollama\n{'='*60}")

# ── Ensure vectorstore is built ───────────────────────────────────────────────
section("0. Vectorstore Check")
if not is_vectorstore_built():
    print("  [INFO] Building vectorstore now...")
    build_vectorstore()
test("Vectorstore is built", is_vectorstore_built())

# ── Test 1: Pricing query ─────────────────────────────────────────────────────
section("1. Pricing Query (intent=sales)")
docs = retrieve("What are your pricing plans?", "sales", top_k=3)
test("Returns list",                  isinstance(docs, list))
test("Returns 3 or fewer results",    0 < len(docs) <= 3, f"got {len(docs)}")
test("Each doc has 'content' key",    all("content" in d for d in docs))
test("Each doc has 'source' key",     all("source" in d for d in docs))
test("Pricing-relevant source found", any("pricing" in d["source"] for d in docs),
     f"Sources: {[d['source'] for d in docs]}")

print(f"\n  Top result from: {docs[0]['source']}")
print(f"  {DIM}{docs[0]['content'][:120]}...{RESET}")

# ── Test 2: Password reset query ──────────────────────────────────────────────
section("2. Password Query (intent=account)")
docs = retrieve("I forgot my password. How do I reset it?", "account", top_k=3)
test("Returns results",               len(docs) > 0, f"got {len(docs)}")
test("Includes policy or faq source", any(d["source"] in ["company_policy.pdf","faq.pdf"] for d in docs),
     f"Sources: {[d['source'] for d in docs]}")
test("Content is non-empty",          all(len(d["content"]) > 20 for d in docs))

print(f"\n  Top result from: {docs[0]['source']}")
print(f"  {DIM}{docs[0]['content'][:120]}...{RESET}")

# ── Test 3: Application crash query ───────────────────────────────────────────
section("3. Crash Query (intent=technical)")
docs = retrieve("My application crashes when I upload a file", "technical", top_k=3)
test("Returns results",               len(docs) > 0, f"got {len(docs)}")
test("Includes technical source",     any("technical" in d["source"] for d in docs),
     f"Sources: {[d['source'] for d in docs]}")
test("Content mentions relevant info", any(
    any(kw in d["content"].lower() for kw in ["upload","file","crash","error","troubleshoot"])
    for d in docs
), f"Contents: {[d['content'][:50] for d in docs]}")

print(f"\n  Top result from: {docs[0]['source']}")
print(f"  {DIM}{docs[0]['content'][:120]}...{RESET}")

# ── Test 4: Refund query ──────────────────────────────────────────────────────
section("4. Refund Query (intent=billing)")
docs = retrieve("I need a refund for my last payment", "billing", top_k=3)
test("Returns results",               len(docs) > 0, f"got {len(docs)}")
test("Includes policy or pricing source", any(
    d["source"] in ["company_policy.pdf","pricing_guide.pdf","faq.pdf"] for d in docs
), f"Sources: {[d['source'] for d in docs]}")
test("Content mentions relevant info", any(
    any(kw in d["content"].lower() for kw in ["refund","billing","payment","policy"])
    for d in docs
))

print(f"\n  Top result from: {docs[0]['source']}")
print(f"  {DIM}{docs[0]['content'][:120]}...{RESET}")

# ── Test 5: memory_recall returns empty ───────────────────────────────────────
section("5. Memory Recall (no RAG expected)")
docs_mem = retrieve("What was my previous issue?", "memory_recall", top_k=3)
test("memory_recall returns empty list", docs_mem == [],
     f"Expected [], got {len(docs_mem)} docs")

# ── Test 6: format_docs_for_prompt ───────────────────────────────────────────
section("6. format_docs_for_prompt()")
pricing_docs = retrieve("What is the cheapest plan?", "sales", top_k=2)
formatted = format_docs_for_prompt(pricing_docs)
test("Returns non-empty string",      isinstance(formatted, str) and len(formatted) > 0)
test("Contains source label",         "[" in formatted and "]" in formatted)
test("format_docs_for_prompt([]) returns empty", format_docs_for_prompt([]) == "")
print(f"\n  Formatted context preview:\n  {DIM}{formatted[:200]}...{RESET}")

# ── Test 7: Scores are present and ordered ────────────────────────────────────
section("7. Scores and ordering")
docs_scored = retrieve("upgrade subscription plan features", "sales", top_k=3)
test("Each doc has 'score' key",      all("score" in d for d in docs_scored))
if len(docs_scored) > 1:
    test("Results ordered by score (ascending)",
         docs_scored[0]["score"] <= docs_scored[-1]["score"])

# ── Summary ───────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'='*60}")
print(f"{BOLD}  RESULTS: {passed}/{total} tests passed{RESET}")
if failed == 0: print(f"  {GREEN}ALL RAG TESTS PASSED{RESET}")
else:           print(f"  {RED}{failed} tests FAILED{RESET}")
print(f"{'='*60}\n")
