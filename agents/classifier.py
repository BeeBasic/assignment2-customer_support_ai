"""
agents/classifier.py — Intent Classification Node
===================================================

PURPOSE
-------
This is the first node the customer query hits in the LangGraph workflow.
It uses the qwen2.5:7b model (via Ollama) to:
  1. Classify the customer's intent into one of five categories
  2. Detect whether the request requires Human-in-the-Loop approval

HOW IT CONNECTS TO THE GRAPH
------------------------------
  START → classify_intent → conditional_router (route_by_intent)

The function `classify_intent` is registered as a LangGraph node.
It receives the full SupportState dict, reads "raw_query", and returns
a partial dict { "intent": ..., "requires_approval": ... } that
LangGraph merges back into the shared state.

DESIGN DECISIONS
-----------------
- format="json" forces the Ollama model to always return valid JSON
- temperature=0 ensures deterministic, reproducible classification
- A keyword-based fallback runs even when JSON parses correctly, to
  double-check the approval flag (belt-and-suspenders approach)
- Invalid intent labels from the LLM are caught and replaced with the
  keyword fallback, so the router never receives an unexpected value
"""

import json
import os
import sys
from typing import Dict, Any, Tuple

# Allow direct execution from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── Constants ────────────────────────────────────────────────────────────────

VALID_INTENTS = frozenset({"sales", "technical", "billing", "account", "memory_recall"})

# Exact phrases that trigger Human-in-the-Loop approval
APPROVAL_TRIGGERS = [
    "refund",
    "subscription cancellation",
    "cancel subscription",
    "cancel my subscription",
    "account closure",
    "close account",
    "close my account",
    "delete account",
    "compensation",
    "compensate",
    "escalation",
    "escalate",
    "speak to manager",
    "speak with manager",
    "talk to manager",
    "manager",
]

# ── System Prompt ─────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """\
You are an intent classification engine for a customer support system.

Analyze the customer query and return a JSON object with EXACTLY these two fields:

{
  "intent": "<one of: sales | technical | billing | account | memory_recall>",
  "requires_approval": <true | false>
}

INTENT DEFINITIONS:
- "sales"         → pricing, subscription plans, upgrades, features, free trial, discounts
- "technical"     → bugs, crashes, errors, how-to guides, setup, integrations, API issues
- "billing"       → invoices, payment failures, refunds, charges, billing disputes
- "account"       → login, password reset, profile changes, subscription cancellation, account closure
- "memory_recall" → customer asking about their previous issue, past support history, prior conversation

REQUIRES_APPROVAL RULES — set to true ONLY when the query contains:
- a refund request
- subscription cancellation
- account closure
- compensation request
- escalation to management

OUTPUT RULES:
- Return ONLY the JSON object. No explanation. No markdown. No extra text.
- The intent must be one of the five exact values listed above (lowercase).
- requires_approval must be boolean true or false (not a string).
"""


# ── Main Node Function ────────────────────────────────────────────────────────

def classify_intent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: classify the customer's intent and detect approval need.

    This function is passed directly to graph.add_node("classifier", classify_intent).
    LangGraph calls it with the full state and merges the returned dict back
    into the running state automatically.

    Args:
        state: The current SupportState containing at minimum "raw_query".

    Returns:
        Dict with two keys that LangGraph merges into state:
          - "intent":            str — one of the five valid intent labels
          - "requires_approval": bool — True if HITL approval is needed
    """
    raw_query: str = state["raw_query"]

    # ── Build LLM ──────────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",   # Forces the model to always respond with valid JSON
        temperature=0,   # Deterministic: same query → same intent every time
    )

    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=f'Customer query: "{raw_query}"'),
    ]

    # ── Call LLM ───────────────────────────────────────────────────────────
    response = llm.invoke(messages)
    raw_output: str = response.content.strip()

    # ── Parse & Validate ───────────────────────────────────────────────────
    intent, requires_approval = _parse_output(raw_output, raw_query)

    print(f"\n[Classifier]")
    print(f"  Query    : {raw_query[:70]}")
    print(f"  Intent   : {intent}")
    print(f"  Approval : {requires_approval}")

    return {
        "intent": intent,
        "requires_approval": requires_approval,
    }


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _parse_output(raw_output: str, original_query: str) -> Tuple[str, bool]:
    """
    Parse the LLM's JSON response into (intent, requires_approval).

    Strategy:
      1. Try to JSON-parse the raw output
      2. Validate that intent is one of the five known labels
      3. Cross-check requires_approval with keyword matching (safety net)
      4. Fall back to full keyword matching if JSON parse fails entirely

    Args:
        raw_output:     The raw string response from the LLM.
        original_query: The customer's original message (for fallback).

    Returns:
        Tuple of (intent: str, requires_approval: bool)
    """
    try:
        data = json.loads(raw_output)
        intent: str = str(data.get("intent", "")).lower().strip()
        requires_approval: bool = bool(data.get("requires_approval", False))

        # Validate intent label — reject hallucinated values
        if intent not in VALID_INTENTS:
            print(f"  [Classifier] Invalid intent '{intent}' from LLM → using keyword fallback")
            intent = _keyword_intent(original_query)

        # Belt-and-suspenders: keyword check always overrides LLM if triggered
        if not requires_approval and _needs_approval(original_query):
            requires_approval = True

        return intent, requires_approval

    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        # LLM returned something un-parseable — fall back entirely to keywords
        print(f"  [Classifier] JSON parse failed → full keyword fallback")
        return _keyword_intent(original_query), _needs_approval(original_query)


def _keyword_intent(query: str) -> str:
    """
    Rule-based intent detection used as a fallback when LLM output is invalid.

    Checks keywords in a priority order that prevents misrouting
    (e.g., "password refund" → billing, not technical).

    Args:
        query: The raw customer query string.

    Returns:
        A valid intent string from VALID_INTENTS.
    """
    q = query.lower()

    # memory_recall first — very specific phrasing
    if any(kw in q for kw in [
        "previous", "last time", "before", "history",
        "earlier", "my issue", "what was", "prior"
    ]):
        return "memory_recall"

    # billing — financial keywords
    if any(kw in q for kw in [
        "refund", "invoice", "payment", "charge", "bill",
        "overcharged", "billing", "money back"
    ]):
        return "billing"

    # sales — commercial intent
    if any(kw in q for kw in [
        "price", "pricing", "plan", "plans", "cost",
        "upgrade", "feature", "trial", "discount", "purchase"
    ]):
        return "sales"

    # technical — error / how-to
    if any(kw in q for kw in [
        "crash", "error", "bug", "broken", "not working",
        "upload", "how to", "setup", "install", "api", "integrate"
    ]):
        return "technical"

    # account — identity / access
    if any(kw in q for kw in [
        "password", "login", "account", "profile",
        "cancel", "close", "delete", "forgot"
    ]):
        return "account"

    # Default: account agent handles general queries
    return "account"


def _needs_approval(query: str) -> bool:
    """
    Check whether the customer query contains any HITL trigger phrases.

    This check runs independently of the LLM so it cannot be bypassed
    by an LLM classification error.

    Args:
        query: The raw customer query string.

    Returns:
        True if any HITL trigger phrase is found; False otherwise.
    """
    q = query.lower()
    return any(trigger in q for trigger in APPROVAL_TRIGGERS)
