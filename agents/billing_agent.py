"""
agents/billing_agent.py — Billing Support Agent Node (with RAG)
================================================================

MODULE 6 UPGRADE: RAG integration added.
- Retrieves from policy_collection + pricing_collection + faq_collection
- Retrieved context injected for grounded, policy-compliant answers
- Stores retrieved docs in state["retrieved_docs"]
"""

import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState
from rag.retriever import retrieve, format_docs_for_prompt

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

BILLING_SYSTEM_PROMPT = """\
You are a professional Billing Support Agent for TechCorp.

Use the COMPANY DOCUMENTATION below to give accurate, policy-compliant answers
about invoices, payments, refunds, and subscription charges.

{rag_context}

FALLBACK KNOWLEDGE (use only if not covered in documentation):
- Refund policy: Full refund within 30 days. After 30 days, case-by-case.
  Processing takes 5-7 business days. ALL refund requests require manager approval.
- Subscription billing: Monthly or annual. Annual saves up to 17%.
  Upgrades: prorated immediately. Downgrades: next billing cycle.
- Payment methods: Visa, Mastercard, American Express, PayPal, Bank Transfer (Enterprise).
- Compensation >$50 requires manager approval.

INSTRUCTIONS:
- Prioritize the COMPANY DOCUMENTATION above.
- For refund or compensation requests: Acknowledge empathetically, confirm you are
  submitting it for manager review, state the 5-7 day timeline.
  Do NOT promise the refund will be approved.
- Keep responses to 3-5 sentences unless detail is needed.
"""


def run_billing_agent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Billing Support Agent with RAG.

    Args:
        state: Current SupportState.

    Returns:
        Dict with "agent_draft" and "retrieved_docs" to merge into state.
    """
    raw_query: str = state["raw_query"]
    requires_approval: bool = state.get("requires_approval", False)

    # ── RAG retrieval ──────────────────────────────────────────────────────
    retrieved_docs = retrieve(raw_query, "billing", top_k=3)
    rag_context = ""
    if retrieved_docs:
        rag_context = "--- COMPANY DOCUMENTATION ---\n"
        rag_context += format_docs_for_prompt(retrieved_docs)
        rag_context += "\n--- END DOCUMENTATION ---\n"
    else:
        rag_context = "(No specific documentation retrieved. Use fallback knowledge.)"

    approval_note = " [REQUIRES APPROVAL]" if requires_approval else ""
    print(f"\n[BillingAgent{approval_note}] RAG: {len(retrieved_docs)} chunks | Query: '{raw_query[:50]}'")

    # ── LLM generation ────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )

    system_prompt = BILLING_SYSTEM_PROMPT.format(rag_context=rag_context)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_query)]

    response = llm.invoke(messages)
    draft: str = response.content.strip()
    print(f"[BillingAgent] Draft generated ({len(draft)} chars)")

    return {"agent_draft": draft, "retrieved_docs": retrieved_docs}
