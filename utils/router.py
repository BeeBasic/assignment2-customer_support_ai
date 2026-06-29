"""
utils/router.py — Conditional Edge Functions for LangGraph
============================================================

PURPOSE
-------
This module contains the two routing functions that LangGraph uses as
conditional edges to decide which node to visit next.

They are PURE PYTHON FUNCTIONS — no LLM calls, no I/O, no side effects.
They simply read State fields and return a node-name string.

HOW LANGGRAPH USES THESE
--------------------------
In graph.py, these functions are passed to add_conditional_edges():

    graph.add_conditional_edges(
        "classifier",       ← source node
        route_by_intent,    ← this function
        {                   ← mapping: return value → node name
            "sales_agent":      "sales_agent",
            "technical_agent":  "technical_agent",
            ...
        }
    )

When LangGraph finishes executing the "classifier" node, it calls
route_by_intent(state) and follows the edge matching the returned string.

FUNCTION SIGNATURES
--------------------
Both functions follow the LangGraph conditional edge contract:
  - Input:  SupportState (the full shared state dict)
  - Output: str (the name of the next node to visit)

NOTE ON approval_router
------------------------
approval_router is implemented here but NOT yet wired into the graph
in this module (Supervisor and HumanApproval nodes aren't built yet).
It will be connected in a later module when those nodes exist.
"""

import os
import sys
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import SupportState

# ── Type aliases for the return values ───────────────────────────────────────
# These match exactly the node names registered in graph.py via add_node().
# If a node name changes in graph.py, update these aliases too.

AgentNodeName = Literal[
    "sales_agent",
    "technical_agent",
    "billing_agent",
    "account_agent",
    "memory_handler",
]

OversightNodeName = Literal[
    "human_approval",
    "supervisor",
]


# ── Router 1: Intent → Agent ──────────────────────────────────────────────────

def route_by_intent(state: SupportState) -> AgentNodeName:
    """
    Conditional edge function: routes from the classifier to the
    correct specialized agent based on state["intent"].

    Called by LangGraph automatically after the "classifier" node runs.
    The return value must exactly match one of the keys in the
    add_conditional_edges() mapping dict in graph.py.

    Routing table:
        "sales"         → "sales_agent"
        "technical"     → "technical_agent"
        "billing"       → "billing_agent"
        "account"       → "account_agent"
        "memory_recall" → "memory_handler"
        <anything else> → "account_agent"  (safe fallback)

    Args:
        state: The current SupportState. Must have "intent" set by the
               classifier node before this function is called.

    Returns:
        The name of the agent node to route to (a string matching a
        registered node name in the LangGraph StateGraph).
    """
    intent: str = state.get("intent") or ""

    route_map: dict[str, AgentNodeName] = {
        "sales":         "sales_agent",
        "technical":     "technical_agent",
        "billing":       "billing_agent",
        "account":       "account_agent",
        "memory_recall": "memory_handler",
    }

    destination = route_map.get(intent.lower(), "account_agent")

    print(f"[Router] intent='{intent}' -> routing to '{destination}'")
    return destination


# ── Router 2: Approval Check → HITL or Supervisor ────────────────────────────

def approval_router(state: SupportState) -> OversightNodeName:
    """
    Conditional edge function: routes from a specialized agent to either
    the Human Approval node (if approval is required) or directly to the
    Supervisor agent.

    This function is NOT yet wired into graph.py in this module.
    It will be connected when the HITL and Supervisor nodes are built.

    Routing logic:
        requires_approval == True  → "human_approval"
        requires_approval == False → "supervisor"

    Args:
        state: The current SupportState. Must have "requires_approval"
               set by the classifier node (or confirmed by the agent).

    Returns:
        "human_approval" if HITL review is needed, else "supervisor".
    """
    requires_approval: bool = state.get("requires_approval", False)

    destination: OversightNodeName = (
        "human_approval" if requires_approval else "supervisor"
    )

    print(f"[ApprovalRouter] requires_approval={requires_approval} -> routing to '{destination}'")
    return destination
