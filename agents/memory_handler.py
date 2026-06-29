"""
agents/memory_handler.py — Memory Recall Handler Node
======================================================

PURPOSE
-------
Handles the "memory_recall" intent — when a customer asks about their
previous conversations, past issues, or support history.

MODULE 5 UPGRADE: REAL SQLITE RETRIEVAL
-----------------------------------------
This replaces the Module 4 stub. The handler now:
  1. Reads state["customer_id"] to identify the customer
  2. Queries SQLite via get_history() to retrieve prior conversation turns
  3. If history exists: passes it to qwen2.5:7b to generate a contextual,
     specific answer to the customer's question
  4. If no history: returns a polite "no history found" message

EXAMPLE CONVERSATION THIS ENABLES
------------------------------------
Turn 1:  Customer: "My name is David."
         Assistant: saves to SQLite

Turn 2:  Customer: "I have a billing issue."
         Assistant: saves to SQLite

Turn 3:  Customer: "What was my previous issue?"
         memory_handler retrieves turns 1 & 2 from SQLite
         LLM generates: "Your previous issue was a billing problem."

HOW IT CONNECTS
----------------
    route_by_intent (intent="memory_recall") -> memory_handler -> END

NOTE: In a later module, memory_handler will go through the Supervisor
before returning the final response.
"""

import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import SupportState
from memory.sqlite_memory import get_history, format_history_for_prompt

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── System prompt for memory recall ──────────────────────────────────────────

MEMORY_RECALL_SYSTEM_PROMPT = """\
You are a helpful customer support agent for TechCorp.

A customer is asking about their previous support interactions.
Below is the customer's conversation history from our records.

CONVERSATION HISTORY:
{history}

INSTRUCTIONS:
- Use ONLY the conversation history above to answer the customer's question.
- Be specific: mention the actual issues, topics, or details discussed.
- If the history shows a specific problem (e.g., billing issue, password reset),
  state it clearly: "Your previous issue was a [specific issue]."
- If you cannot find a relevant answer in the history, say so honestly.
- Keep your response to 2-4 sentences.
- Do NOT invent details that are not in the conversation history.
"""


def run_memory_handler(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Memory Recall Handler.

    Retrieves the customer's conversation history from SQLite and uses
    the LLM to generate a contextual answer to their recall question.

    Workflow:
      1. Extract customer_id from state
      2. Query SQLite for last 10 turns via get_history()
      3. If history found: inject into prompt and call LLM
      4. If no history: return a polite "nothing found" response

    Args:
        state: Current SupportState. Reads customer_id and raw_query.

    Returns:
        Dict with "agent_draft" set to the memory recall response.
    """
    customer_id: str = state.get("customer_id", "unknown")
    raw_query: str = state.get("raw_query", "")

    print(f"\n[MemoryHandler] Retrieving history for customer '{customer_id}'")

    # ── Step 1: Retrieve history from SQLite ───────────────────────────────
    history = get_history(customer_id, limit=20)

    if not history:
        print(f"[MemoryHandler] No history found for '{customer_id}'")
        draft = (
            "I checked our records but couldn't find any previous conversations "
            "linked to your account. This may be your first interaction with us, "
            "or your history may have been cleared. Please describe your current "
            "issue and I'll be happy to help you right away."
        )
        return {"agent_draft": draft}

    # ── Step 2: Format history for the LLM ────────────────────────────────
    formatted_history = format_history_for_prompt(history)
    print(f"[MemoryHandler] Found {len(history)} turns. Generating contextual response.")
    print(f"[MemoryHandler] History preview: {formatted_history[:120]}...")

    # ── Step 3: Build prompt with history injected ─────────────────────────
    system_prompt = MEMORY_RECALL_SYSTEM_PROMPT.format(history=formatted_history)

    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,  # Low: recall answers must be factually grounded in history
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=raw_query),
    ]

    # ── Step 4: Generate response ─────────────────────────────────────────
    response = llm.invoke(messages)
    draft: str = response.content.strip()

    print(f"[MemoryHandler] Draft generated ({len(draft)} chars)")
    return {"agent_draft": draft}
