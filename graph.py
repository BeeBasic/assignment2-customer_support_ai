"""
graph.py — LangGraph StateGraph Definition
===========================================

PURPOSE
-------
This module builds and compiles the complete LangGraph workflow for the
Customer Support AI system. It wires every node and every edge together
into a compiled, executable graph object.

GRAPH STRUCTURE (Module 6-8)
-----------------------------
The complete production workflow:

    START
      |
      v
  [classifier]
      |
      v (conditional: route_by_intent)
  ┌───┴───────────────────────────────────┐
  v           v            v        v     v
[sales]  [technical]  [billing] [account] [memory]
  │           │            │        │         │
  └───────────┴────────────┼────────┴─────────┘
                           │ (conditional: approval_router)
                    ┌──────┴──────┐
                    ▼             ▼
             [human_approval] [supervisor]
                    │             │
        (post_approval_router)    │
              ┌─────┴─────┐       │
              ▼           ▼       ▼
             END     [supervisor_agent]
                            │
                            ▼
                           END

HOW TO USE
-----------
    from graph import build_graph
    graph = build_graph()
    result = graph.invoke(state, config={"configurable": {"thread_id": "..."}})
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import SupportState

# ── Node functions ─────────────────────────────────────────────────────────
from agents.classifier import classify_intent
from agents.sales_agent import run_sales_agent
from agents.technical_agent import run_technical_agent
from agents.billing_agent import run_billing_agent
from agents.account_agent import run_account_agent
from agents.memory_handler import run_memory_handler
from agents.supervisor_agent import run_supervisor_agent
from utils.approval import human_approval_node

# ── Routing functions (conditional edges) ──────────────────────────────────
from utils.router import route_by_intent, approval_router
from utils.approval import post_approval_router

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Persistent SQLite Connection for Checkpointer ──────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DB = os.path.join(_PROJECT_ROOT, "database", "checkpoints.db")
os.makedirs(os.path.dirname(CHECKPOINT_DB), exist_ok=True)

# check_same_thread=False allows sharing between main and sub threads safely
_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph():
    """
    Construct and compile the LangGraph StateGraph for the support system.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for invoke() / stream()
        with built-in SqliteSaver checkpointing.
    """
    builder = StateGraph(SupportState)

    # ── 1. Register nodes ──────────────────────────────────────────────────
    builder.add_node("classifier",        classify_intent)
    builder.add_node("sales_agent",       run_sales_agent)
    builder.add_node("technical_agent",   run_technical_agent)
    builder.add_node("billing_agent",     run_billing_agent)
    builder.add_node("account_agent",     run_account_agent)
    builder.add_node("memory_handler",    run_memory_handler)
    builder.add_node("human_approval",    human_approval_node)
    builder.add_node("supervisor",        run_supervisor_agent)

    # ── 2. Entry edge ─────────────────────────────────────────────────────
    builder.add_edge(START, "classifier")

    # ── 3. Intent routing conditional edge ───────────────────────────────
    builder.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "sales_agent":      "sales_agent",
            "technical_agent":  "technical_agent",
            "billing_agent":    "billing_agent",
            "account_agent":    "account_agent",
            "memory_handler":   "memory_handler",
        }
    )

    # ── 4. Approval check routing from every agent node ───────────────────
    for agent_node in ["sales_agent", "technical_agent", "billing_agent", "account_agent", "memory_handler"]:
        builder.add_conditional_edges(
            agent_node,
            approval_router,
            {
                "human_approval": "human_approval",
                "supervisor":     "supervisor",
            }
        )

    # ── 5. Human approval outcomes ────────────────────────────────────────
    builder.add_conditional_edges(
        "human_approval",
        post_approval_router,
        {
            "supervisor": "supervisor",
            "end":        END,
        }
    )

    # ── 6. Supervisor quality checkpoint ──────────────────────────────────
    builder.add_edge("supervisor", END)

    # ── 7. Compile with SqliteSaver Checkpointer ──────────────────────────
    graph = builder.compile(checkpointer=_checkpointer)

    print("[Graph] StateGraph compiled successfully with SqliteSaver.")
    return graph


if __name__ == "__main__":
    g = build_graph()
    print("\nMermaid diagram:")
    print(g.get_graph().draw_mermaid())

