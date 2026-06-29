"""
agents/supervisor_agent.py — Supervisor Quality Control Node
=============================================================

PURPOSE
-------
The Supervisor is the final quality gate before a response reaches the customer.
It reviews the specialized agent's draft against the retrieved documentation and
checks four dimensions:

  1. FACTUAL CORRECTNESS  — Does the draft accurately reflect the documents?
  2. PROFESSIONALISM      — Is the tone appropriate, grammatically correct?
  3. COMPLETENESS         — Does it fully answer the customer's question?
  4. POLICY COMPLIANCE    — Does it comply with TechCorp's policies?

If the draft passes all checks: Supervisor makes minor polish edits.
If the draft fails any check: Supervisor rewrites the response from scratch.

The final output is written to state["final_response"] — this is what the
customer actually receives.

HOW IT CONNECTS
----------------
                [any agent]
                    |
            approval_router()
                /        \
    human_approval        |
         |                |
  post_approval_router    |
         \               /
          [supervisor_agent]
                 |
                END
"""

import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState
from rag.retriever import format_docs_for_prompt

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

SUPERVISOR_SYSTEM_PROMPT = """\
You are a Senior Customer Support Quality Manager at TechCorp.

Your job is to review a support agent's draft response and ensure it meets
our quality standards before it is sent to the customer.

ORIGINAL CUSTOMER QUERY:
{raw_query}

AGENT DRAFT RESPONSE:
{agent_draft}

{doc_section}

REVIEW CHECKLIST — verify all four dimensions:

1. FACTUAL CORRECTNESS
   - Does the draft match the documentation provided?
   - Are pricing figures, timelines, and policies accurate?
   - Flag: any number, date, or policy claim not supported by documentation

2. PROFESSIONALISM
   - Is the tone friendly, empathetic, and professional?
   - Is grammar correct? Is it free of jargon or dismissive language?
   - Flag: robotic tone, typos, overly formal or casual language

3. COMPLETENESS
   - Does the draft fully answer the customer's specific question?
   - Does it provide next steps or a call-to-action?
   - Flag: missing key information, vague or non-committal answers

4. POLICY COMPLIANCE
   - Does the draft comply with TechCorp's policies?
   - Are approvals mentioned where required (refunds, cancellations)?
   - Flag: promising unauthorized discounts, bypassing approval requirements

DECISION:
- If the draft passes all four checks: make only minor grammar/style edits.
- If the draft fails any check: rewrite the response to be accurate and complete.

OUTPUT RULES:
- Return ONLY the final customer-facing response text.
- Do NOT include your review notes, checklist, or meta-commentary.
- Do NOT prefix with "Here is the revised response:" or similar.
- The response should be ready to copy-paste to the customer.
"""


def run_supervisor_agent(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Supervisor Quality Control Agent.

    Reviews the agent draft against retrieved documentation and produces
    the final customer-facing response.

    Args:
        state: Current SupportState. Reads:
          - raw_query:      Original customer question
          - agent_draft:    Draft from specialized agent (or edited by human)
          - retrieved_docs: RAG chunks for fact-checking (may be None)
          - approval_status: Whether HITL approved/edited/bypassed

    Returns:
        Dict with "final_response" to merge into state.
    """
    raw_query    = state.get("raw_query", "")
    agent_draft  = state.get("agent_draft", "")
    retrieved_docs = state.get("retrieved_docs") or []
    approval_status = state.get("approval_status", None)

    # ── Handle rejection (should not normally reach here) ──────────────────
    if approval_status == "rejected":
        # Human rejected — return the pre-set rejection response
        final = state.get("final_response", "We cannot process this request at this time.")
        print(f"[Supervisor] Request was rejected by human reviewer.")
        return {"final_response": final}

    # ── Handle human edit override ─────────────────────────────────────────
    if approval_status == "edited":
        print(f"[Supervisor] Draft was explicitly edited by human reviewer. Bypassing review to respect human override.")
        return {"final_response": agent_draft}

    # ── Guard: empty draft ─────────────────────────────────────────────────
    if not agent_draft:
        print(f"[Supervisor] WARNING: No agent draft found. Generating fallback.")
        return {
            "final_response": (
                "Thank you for contacting TechCorp support. I apologize for the "
                "inconvenience — please try again or contact us at support@techcorp.com."
            )
        }

    # ── Build documentation context ───────────────────────────────────────
    doc_section = ""
    if retrieved_docs:
        formatted = format_docs_for_prompt(retrieved_docs)
        doc_section = (
            "RETRIEVED DOCUMENTATION (use for fact-checking):\n"
            f"{formatted}\n"
        )
    else:
        doc_section = "(No retrieved documentation available for this response type.)"

    print(f"\n[Supervisor] Reviewing draft ({len(agent_draft)} chars) "
          f"with {len(retrieved_docs)} docs | approval_status={approval_status}")

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        raw_query=raw_query,
        agent_draft=agent_draft,
        doc_section=doc_section,
    )

    # ── LLM review ────────────────────────────────────────────────────────
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,  # Low: consistency is more important than creativity here
    )

    response = llm.invoke([SystemMessage(content=prompt)])
    final_response: str = response.content.strip()

    print(f"[Supervisor] Final response generated ({len(final_response)} chars)")

    return {"final_response": final_response}
