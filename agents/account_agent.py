"""
agents/account_agent.py — Account Support Agent Node (with RAG)
================================================================

MODULE 6 UPGRADE: RAG integration added.
- Retrieves from policy_collection + faq_collection
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

ACCOUNT_SYSTEM_PROMPT = """\
You are a helpful Account Support Agent for TechCorp.

Use the COMPANY DOCUMENTATION below to give accurate, policy-compliant answers
about account management, login, password resets, and account changes.

{rag_context}

FALLBACK KNOWLEDGE (use only if not in documentation):
- Password reset: app.techcorp.com/reset-password — link expires in 24 hours.
  Requirements: 8+ chars, uppercase, number, symbol.
- Subscription cancellation: takes effect at end of billing period.
  Requires supervisor approval (24-hour processing).
- Account closure: permanent, all data deleted within 30 days (GDPR).
  Customer must export data first via Settings > Account > Export.
  Requires senior support approval. Cannot be undone.
- Data export: Settings > Account > Export Data (JSON, CSV, XML). Up to 24 hours.
- SSO: Enterprise plans only (SAML 2.0, OAuth 2.0, OpenID Connect).

INSTRUCTIONS:
- Prioritize the COMPANY DOCUMENTATION above.
- For cancellation or account closure: Acknowledge, note approval requirement,
  warn about consequences, offer alternatives (pause, downgrade).
- For password issues: Give clear step-by-step instructions.
- Be warm and professional.
"""


def run_account_agent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Account Support Agent with RAG.

    Args:
        state: Current SupportState.

    Returns:
        Dict with "agent_draft" and "retrieved_docs" to merge into state.
    """
    raw_query: str = state["raw_query"]
    requires_approval: bool = state.get("requires_approval", False)

    # ── RAG retrieval ──────────────────────────────────────────────────────
    retrieved_docs = retrieve(raw_query, "account", top_k=3)
    rag_context = ""
    if retrieved_docs:
        rag_context = "--- COMPANY DOCUMENTATION ---\n"
        rag_context += format_docs_for_prompt(retrieved_docs)
        rag_context += "\n--- END DOCUMENTATION ---\n"
    else:
        rag_context = "(No specific documentation retrieved. Use fallback knowledge.)"

    approval_note = " [REQUIRES APPROVAL]" if requires_approval else ""
    print(f"\n[AccountAgent{approval_note}] RAG: {len(retrieved_docs)} chunks | Query: '{raw_query[:50]}'")

    # ── LLM generation ────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.2,
    )

    system_prompt = ACCOUNT_SYSTEM_PROMPT.format(rag_context=rag_context)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_query)]

    response = llm.invoke(messages)
    draft: str = response.content.strip()
    print(f"[AccountAgent] Draft generated ({len(draft)} chars)")

    return {"agent_draft": draft, "retrieved_docs": retrieved_docs}
