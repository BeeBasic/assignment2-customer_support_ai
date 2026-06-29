"""
utils/approval.py — Human-in-the-Loop (HITL) Approval Node
============================================================

PURPOSE
-------
Implements the Human-in-the-Loop approval mechanism for sensitive requests:
  - Refund requests
  - Subscription cancellations
  - Account closures
  - Compensation requests
  - Escalations to management

HOW LANGGRAPH INTERRUPT() WORKS
---------------------------------
1. human_approval_node() calls interrupt(approval_data) — this:
   a. Saves the current state to the SqliteSaver checkpoint
   b. Pauses graph execution
   c. Returns control to the caller of graph.invoke()

2. The caller sees that graph.get_state(config).next is non-empty
   (the graph hasn't reached END yet)

3. The human reviews the draft and enters a decision

4. The caller resumes execution with:
   graph.invoke(Command(resume=decision_dict), config=config)

5. interrupt() returns the decision_dict to the node
   (the code below the interrupt() call executes)

6. Execution continues through post_approval_router -> supervisor -> END

APPROVAL DECISIONS
-------------------
  {"decision": "approved", "feedback": ""}
    -> Draft proceeds to Supervisor unchanged

  {"decision": "edited",   "feedback": "<new draft text>"}
    -> feedback becomes the new agent_draft, sent to Supervisor

  {"decision": "rejected", "feedback": "<reason>"}
    -> Rejection message set as final_response, graph goes to END
"""

import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.types import interrupt
from state import SupportState


# Track sessions that have already printed the pause banner to prevent duplicates on resume
_printed_sessions = set()


# ── Human Approval Node ───────────────────────────────────────────────────────

def human_approval_node(state: SupportState) -> Dict[str, Any]:
    """
    LangGraph node: Human-in-the-Loop approval checkpoint.

    Pauses graph execution and waits for a human reviewer to approve,
    reject, or edit the agent's draft response before it goes to the
    Supervisor.

    The interrupt() call saves state to checkpoint and pauses execution.
    The graph resumes when graph.invoke(Command(resume=value)) is called.

    Args:
        state: Current SupportState. Reads:
          - customer_id, raw_query, intent, agent_draft, requires_approval

    Returns:
        Dict updating state with approval_status and optionally
        human_feedback or final_response (for rejections).
    """
    session_id = state.get("session_id", "unknown")

    # ── Package approval request data ─────────────────────────────────────
    approval_request = {
        "customer_id":   state.get("customer_id", "unknown"),
        "raw_query":     state.get("raw_query", ""),
        "intent":        state.get("intent", "unknown"),
        "agent_draft":   state.get("agent_draft", ""),
        "message": (
            f"APPROVAL REQUIRED\n"
            f"Customer ID : {state.get('customer_id')}\n"
            f"Intent      : {state.get('intent')}\n"
            f"Query       : {state.get('raw_query', '')}\n"
            f"Draft       : {state.get('agent_draft', '')[:200]}"
        ),
    }

    if session_id not in _printed_sessions:
        print(f"\n[HumanApproval] *** PAUSING FOR HUMAN REVIEW ***")
        print(f"  Customer : {approval_request['customer_id']}")
        print(f"  Intent   : {approval_request['intent']}")
        print(f"  Query    : {approval_request['raw_query']}")
        print(f"  Draft    : {approval_request['agent_draft'][:120]}...")
        print(f"[HumanApproval] Calling interrupt() — awaiting human decision...")
        _printed_sessions.add(session_id)

    # ── PAUSE EXECUTION HERE ───────────────────────────────────────────────
    # interrupt() saves state to checkpoint and returns control to caller.
    # Execution resumes at this exact point when Command(resume=...) is called.
    human_decision: Dict = interrupt(approval_request)
    # ── RESUMED HERE ───────────────────────────────────────────────────────

    decision = human_decision.get("decision", "approved").lower().strip()
    feedback = human_decision.get("feedback", "").strip()

    print(f"[HumanApproval] Decision received: '{decision}' | feedback: '{feedback[:80]}'")

    # ── Handle decision outcomes ──────────────────────────────────────────
    if decision == "rejected":
        rejection_msg = (
            "We have reviewed your request carefully. Unfortunately, we are unable "
            "to approve this request at this time. "
            + (f"Reason: {feedback}" if feedback else
               "Please contact our support team for further assistance.")
        )
        return {
            "approval_status": "rejected",
            "human_feedback":  feedback,
            "final_response":  rejection_msg,
        }

    elif decision == "edited":
        if not feedback:
            print("[HumanApproval] WARNING: 'edited' chosen but no feedback provided. Treating as approved.")
            return {
                "approval_status": "approved",
                "human_feedback":  "",
            }
        return {
            "approval_status": "edited",
            "human_feedback":  feedback,
            "agent_draft":     feedback,  # Replace draft with human's version
        }

    else:  # "approved" or anything else
        return {
            "approval_status": "approved",
            "human_feedback":  feedback,
        }


# ── Post-Approval Conditional Router ─────────────────────────────────────────

def post_approval_router(state: SupportState) -> str:
    """
    Conditional edge function: routes from human_approval_node to either
    the Supervisor (for approved/edited requests) or END (for rejections).

    Called by LangGraph after human_approval_node completes.

    Args:
        state: Current SupportState. Reads approval_status.

    Returns:
        "supervisor" if approved or edited.
        "end"        if rejected (final_response is already set).
    """
    approval_status = state.get("approval_status", "approved")

    if approval_status == "rejected":
        print(f"[PostApprovalRouter] Rejected -> routing to END")
        return "end"

    print(f"[PostApprovalRouter] Status='{approval_status}' -> routing to supervisor")
    return "supervisor"
