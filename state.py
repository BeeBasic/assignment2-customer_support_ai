"""
state.py — Shared State Definition for the Customer Support AI Graph
======================================================================

PURPOSE
-------
This module defines the single source of truth that flows through every
node in the LangGraph workflow. Every node receives the full State object,
reads the fields it cares about, updates only its own fields, and returns
the modified State to LangGraph.

LangGraph merges each node's returned dict back into the global State
automatically — so nodes only need to return the fields they changed.

HOW STATE FLOWS THROUGH THE GRAPH
-----------------------------------
customer_query     → writes: raw_query, customer_id, session_id, metadata
intent_classifier  → writes: intent, requires_approval
memory_retriever   → writes: conversation_history
rag_retriever      → writes: retrieved_docs
[agent node]       → writes: agent_draft (also may confirm requires_approval)
human_approval     → writes: approval_status, human_feedback
supervisor_agent   → writes: final_response
final_response_node→ writes: nothing new (reads final_response, saves to DB)

WHY TypedDict?
--------------
LangGraph requires State to be defined as a TypedDict (or Pydantic model).
TypedDict gives us:
  - Type hints for every field (IDE support, fewer bugs)
  - Dictionary-style access (state["intent"])
  - Compatibility with LangGraph's internal merge logic

DESIGN DECISION: Optional Fields
---------------------------------
Most fields start as None and get populated as the graph executes.
This is intentional — the graph doesn't know at startup which agent
will run, whether HITL is needed, etc. Fields are filled lazily.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


# ── Conversation History Entry ───────────────────────────────────────────────
# Each turn in the customer's conversation history is stored as a dict
# with 'role' ("user" or "assistant") and 'content' (the message text).
# This mirrors the standard OpenAI / LangChain message format.
ConversationTurn = Dict[str, str]

# ── Document Chunk (from RAG retrieval) ──────────────────────────────────────
# Each retrieved chunk is a dict with 'content' (the text) and 'source'
# (the document filename it came from, e.g., "pricing_guide.pdf").
DocumentChunk = Dict[str, str]


class SupportState(TypedDict):
    """
    The complete shared state object for the Customer Support AI graph.

    This TypedDict is passed to every node as input and returned (partially)
    by every node as output. LangGraph merges the returned fields into the
    running state automatically.

    Fields are grouped by the node that is responsible for writing them.
    """

    # ── Set by: customer_query node (entry point) ────────────────────────────

    customer_id: str
    """
    Unique identifier for the customer making the request.
    Used to retrieve conversation history from SQLite and to tag
    new conversation turns when saving to the database.
    Example: "CUST-001", "user_abc123"
    """

    session_id: str
    """
    Unique identifier for the current conversation session.
    Distinguishes multiple sessions from the same customer.
    Also used as LangGraph's thread_id for checkpoint storage,
    enabling HITL pause/resume to work correctly.
    Example: "sess-2024-06-28-001"
    """

    raw_query: str
    """
    The original, unmodified message from the customer.
    This is never altered after being set. All downstream nodes
    work with this field to understand what the customer asked.
    Example: "I need a refund for my last invoice."
    """

    metadata: Dict[str, Any]
    """
    Supplementary information about the request.
    Populated at entry and can be enriched by later nodes.
    Typical keys:
      - "timestamp": ISO format datetime when the query arrived
      - "model_used": which LLM model generated the response
      - "total_tokens": token count (if available)
      - "processing_time_ms": end-to-end latency
    """

    # ── Set by: intent_classifier node ──────────────────────────────────────

    intent: Optional[str]
    """
    The classified intent of the customer query.
    Set exclusively by the intent_classifier node.
    Used by the conditional_router to decide which agent to invoke.

    Valid values:
      - "sales"        → route to Sales Agent
      - "technical"    → route to Technical Support Agent
      - "billing"      → route to Billing Agent
      - "account"      → route to Account Agent
      - "memory_recall"→ bypass agents, go directly to SQLite memory
    """

    requires_approval: bool
    """
    Flag indicating whether this request must go through Human-in-the-Loop
    approval before the Supervisor generates the final response.

    Set to True by the intent_classifier when the query contains any of:
      - "refund"
      - "subscription cancellation"
      - "account closure"
      - "compensation"
      - "escalation to management"

    When True, LangGraph routes through human_approval_node (which pauses
    execution via interrupt()) before proceeding to the supervisor.
    """

    # ── Set by: memory_retriever node ────────────────────────────────────────

    conversation_history: Optional[List[ConversationTurn]]
    """
    The last N conversation turns for this customer, retrieved from SQLite.
    Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Injected into the agent's prompt so the agent can answer questions
    like "What was my previous issue?" with real context.

    None if no prior history exists (first-time customer).
    """

    # ── Set by: rag_retriever node ────────────────────────────────────────────

    retrieved_docs: Optional[List[DocumentChunk]]
    """
    Top-K document chunks retrieved from ChromaDB, ranked by cosine
    similarity to the customer's query.
    Format: [{"content": "...", "source": "pricing_guide.pdf"}, ...]

    Each specialized agent retrieves from its own relevant collections.
    The Supervisor Agent also reads this field to verify factual accuracy.

    None if RAG retrieval was skipped (e.g., memory_recall intent).
    """

    # ── Set by: specialized agent nodes ──────────────────────────────────────

    agent_draft: Optional[str]
    """
    The raw response draft generated by the specialized agent
    (sales_agent, technical_agent, billing_agent, or account_agent).

    This draft has NOT been reviewed by the Supervisor yet.
    If requires_approval is True, the human reviewer also sees this draft
    before it proceeds to the Supervisor.

    The Supervisor may rewrite, improve, or reject this draft entirely.
    """

    # ── Set by: human_approval_node ──────────────────────────────────────────

    approval_status: Optional[str]
    """
    The outcome of the Human-in-the-Loop review.
    Only meaningful when requires_approval is True.

    Valid values:
      - None       → HITL not yet triggered or not required
      - "pending"  → Waiting for human reviewer (graph is paused)
      - "approved" → Human approved the draft, continue to Supervisor
      - "rejected" → Human rejected the request, send rejection message
      - "edited"   → Human edited the draft before approving

    LangGraph resumes from the checkpoint when this changes from "pending".
    """

    human_feedback: Optional[str]
    """
    Optional text from the human reviewer, set alongside approval_status.

    Use cases:
      - "approved" with no feedback → Supervisor uses agent_draft as-is
      - "edited" with feedback → The edited version of the draft
      - "rejected" with feedback → The rejection reason to explain to customer

    Example: "Please also mention that the refund will take 5-7 business days."
    """

    # ── Set by: supervisor_agent node ─────────────────────────────────────────

    final_response: Optional[str]
    """
    The verified, polished response produced by the Supervisor Agent.
    This is the text that will actually be sent to the customer.

    The Supervisor checks the agent_draft against:
      1. Factual correctness  (cross-referenced with retrieved_docs)
      2. Policy compliance    (cross-referenced with company_policy.pdf)
      3. Professionalism      (tone, grammar, clarity)
      4. Completeness         (does it fully answer the question?)

    If the draft passes all checks: final_response ≈ agent_draft (minor edits)
    If the draft fails:            final_response is a Supervisor rewrite
    """


def create_initial_state(
    customer_id: str,
    raw_query: str,
    session_id: Optional[str] = None
) -> SupportState:
    """
    Factory function to create a fresh SupportState at the start of each
    customer interaction.

    Called by app.py before invoking the LangGraph graph. Ensures all
    optional fields are initialized to None (not missing) so every node
    can safely read any field without a KeyError.

    Args:
        customer_id: The unique ID of the customer making the request.
        raw_query:   The raw text message from the customer.
        session_id:  Optional session ID. If not provided, a unique ID is
                     generated from customer_id + current timestamp.

    Returns:
        A fully initialized SupportState dict ready to pass to graph.invoke().

    Example:
        state = create_initial_state(
            customer_id="CUST-001",
            raw_query="I need a refund for my last payment.",
        )
        result = graph.invoke(state, config={"configurable": {"thread_id": state["session_id"]}})
    """
    # Generate a session ID if one wasn't provided.
    # Format: "CUST-001-20240628-180628"
    # This also serves as the LangGraph thread_id for HITL checkpoint/resume.
    if session_id is None:
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_id = f"{customer_id}-{timestamp_str}"

    return SupportState(
        # ── Entry fields ─────────────────────────────────────────────────────
        customer_id=customer_id,
        session_id=session_id,
        raw_query=raw_query,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "model_used": None,         # Filled by agent nodes
            "processing_time_ms": None  # Filled by final_response_node
        },

        # ── Classification fields ─────────────────────────────────────────────
        intent=None,             # Filled by intent_classifier
        requires_approval=False, # Overridden by intent_classifier if triggered

        # ── Memory & RAG fields ───────────────────────────────────────────────
        conversation_history=None,   # Filled by memory_retriever
        retrieved_docs=None,         # Filled by rag_retriever

        # ── Generation fields ─────────────────────────────────────────────────
        agent_draft=None,       # Filled by specialized agent

        # ── HITL fields ────────────────────────────────────────────────────────
        approval_status=None,   # Filled by human_approval_node
        human_feedback=None,    # Filled by human reviewer during HITL

        # ── Output fields ──────────────────────────────────────────────────────
        final_response=None,    # Filled by supervisor_agent
    )


def format_state_summary(state: SupportState) -> str:
    """
    Returns a human-readable summary of the current State for debugging.

    Call this at any point in development to inspect what the graph knows
    at that stage of execution.

    Args:
        state: The current SupportState dict.

    Returns:
        A formatted multi-line string summarising key state fields.

    Example:
        print(format_state_summary(state))
    """
    history_count = len(state.get("conversation_history") or [])
    docs_count = len(state.get("retrieved_docs") or [])

    return (
        f"\n{'='*60}\n"
        f"  STATE SUMMARY\n"
        f"{'='*60}\n"
        f"  Customer ID   : {state.get('customer_id', 'N/A')}\n"
        f"  Session ID    : {state.get('session_id', 'N/A')}\n"
        f"  Raw Query     : {state.get('raw_query', 'N/A')[:80]}...\n"
        f"  Intent        : {state.get('intent', 'Not classified yet')}\n"
        f"  Needs Approval: {state.get('requires_approval', False)}\n"
        f"  Approval Status: {state.get('approval_status', 'N/A')}\n"
        f"  History Turns : {history_count} turns retrieved\n"
        f"  RAG Docs      : {docs_count} chunks retrieved\n"
        f"  Has Draft     : {'Yes' if state.get('agent_draft') else 'No'}\n"
        f"  Has Response  : {'Yes' if state.get('final_response') else 'No'}\n"
        f"{'='*60}\n"
    )
