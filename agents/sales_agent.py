"""
agents/sales_agent.py — Sales Support Agent Node (with RAG)
============================================================

MODULE 6 UPGRADE: RAG integration added.
- Retrieves from pricing_collection + faq_collection before generating
- Retrieved context injected into system prompt for grounded answers
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

SALES_SYSTEM_PROMPT = """\
You are a friendly and knowledgeable Sales Support Agent for TechCorp.

Your role is to help customers understand our subscription plans, pricing,
features, and upgrade options. Use the COMPANY DOCUMENTATION below to give
accurate, specific answers.

{rag_context}

FALLBACK KNOWLEDGE (use only if documentation does not cover the topic):
- Starter Plan: $9/month - up to 3 users, 10 GB storage, email support
- Professional Plan: $29/month - up to 25 users, 100 GB storage, priority support, API access
- Enterprise Plan: Custom pricing - unlimited users, dedicated manager, SSO
- Annual billing saves up to 17%
- Non-profits get 30% discount, educational institutions get 50%

INSTRUCTIONS:
- Prioritize information from the COMPANY DOCUMENTATION above.
- Be concise but complete. Answer in 3-5 sentences.
- Always mention the free trial if the customer seems undecided.
- Do NOT make up pricing or features not in the documentation.
"""


def run_sales_agent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Sales Support Agent with RAG.

    Retrieves relevant pricing/FAQ chunks, injects into prompt,
    and generates a grounded sales response.

    Args:
        state: Current SupportState.

    Returns:
        Dict with "agent_draft" and "retrieved_docs" to merge into state.
    """
    raw_query: str = state["raw_query"]

    # ── RAG retrieval ──────────────────────────────────────────────────────
    retrieved_docs = retrieve(raw_query, "sales", top_k=3)
    rag_context = ""
    if retrieved_docs:
        rag_context = "--- COMPANY DOCUMENTATION ---\n"
        rag_context += format_docs_for_prompt(retrieved_docs)
        rag_context += "\n--- END DOCUMENTATION ---\n"
    else:
        rag_context = "(No specific documentation retrieved. Use fallback knowledge.)"

    print(f"\n[SalesAgent] RAG: {len(retrieved_docs)} chunks | Query: '{raw_query[:50]}'")

    # ── LLM generation ────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.3,
    )

    system_prompt = SALES_SYSTEM_PROMPT.format(rag_context=rag_context)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_query)]

    response = llm.invoke(messages)
    draft: str = response.content.strip()
    print(f"[SalesAgent] Draft generated ({len(draft)} chars)")

    return {"agent_draft": draft, "retrieved_docs": retrieved_docs}
