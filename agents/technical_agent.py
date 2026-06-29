"""
agents/technical_agent.py — Technical Support Agent Node (with RAG)
====================================================================

MODULE 6 UPGRADE: RAG integration added.
- Retrieves from tech_collection + faq_collection before generating
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

TECHNICAL_SYSTEM_PROMPT = """\
You are an expert Technical Support Agent for TechCorp.

Use the COMPANY DOCUMENTATION below to give accurate, step-by-step solutions.
The documentation contains error codes, troubleshooting steps, and system specs.

{rag_context}

FALLBACK KNOWLEDGE (use only if not covered in documentation):
- File upload crashes: check file size limits (Starter: 25MB, Pro: 100MB, Enterprise: 500MB)
  Supported: PDF, DOCX, XLSX, PPTX, PNG, JPG, GIF, MP4, ZIP
  Fix: clear browser cache, try different browser, use desktop app for large files
- Password reset: app.techcorp.com/reset-password — link expires in 24 hours
- API errors: 401=expired token (regenerate in Settings>API), 429=rate limit exceeded
- System requirements: Windows 10+/macOS 11+/Ubuntu 20.04+, Chrome 90+, 4GB RAM

INSTRUCTIONS:
- Prioritize the COMPANY DOCUMENTATION above over fallback knowledge.
- Give numbered, step-by-step instructions when solving a problem.
- Be specific. Avoid generic "contact support" unless truly necessary.
- Keep responses focused: 3-6 sentences or numbered steps.
"""


def run_technical_agent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Technical Support Agent with RAG.

    Retrieves relevant tech manual + FAQ chunks, injects into prompt,
    and generates a grounded technical response.

    Args:
        state: Current SupportState.

    Returns:
        Dict with "agent_draft" and "retrieved_docs" to merge into state.
    """
    raw_query: str = state["raw_query"]

    # ── RAG retrieval ──────────────────────────────────────────────────────
    retrieved_docs = retrieve(raw_query, "technical", top_k=3)
    rag_context = ""
    if retrieved_docs:
        rag_context = "--- COMPANY DOCUMENTATION ---\n"
        rag_context += format_docs_for_prompt(retrieved_docs)
        rag_context += "\n--- END DOCUMENTATION ---\n"
    else:
        rag_context = "(No specific documentation retrieved. Use fallback knowledge.)"

    print(f"\n[TechnicalAgent] RAG: {len(retrieved_docs)} chunks | Query: '{raw_query[:50]}'")

    # ── LLM generation ────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )

    system_prompt = TECHNICAL_SYSTEM_PROMPT.format(rag_context=rag_context)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=raw_query)]

    response = llm.invoke(messages)
    draft: str = response.content.strip()
    print(f"[TechnicalAgent] Draft generated ({len(draft)} chars)")

    return {"agent_draft": draft, "retrieved_docs": retrieved_docs}
