"""
memory/sqlite_memory.py — Persistent SQLite Conversation Memory
================================================================

PURPOSE
-------
Provides a simple, lightweight conversation memory layer using Python's
built-in sqlite3 module. Stores every customer support interaction so that:

  1. The memory_handler can retrieve previous conversations when a customer
     asks "What was my previous issue?" or "What did we discuss?"

  2. Future agents will have prior context injected into their prompts
     (full RAG + memory integration in later modules)

  3. The graph has a persistent audit log of all interactions

DATABASE SCHEMA
----------------
Table: conversations
  id          INTEGER  PRIMARY KEY AUTOINCREMENT
  customer_id TEXT     NOT NULL  — who sent the message
  session_id  TEXT     NOT NULL  — which session it belongs to
  role        TEXT     NOT NULL  — "user" or "assistant"
  message     TEXT     NOT NULL  — the actual text content
  timestamp   TEXT     NOT NULL  — ISO 8601 datetime string

Index: idx_customer_timestamp
  ON conversations (customer_id, timestamp)
  — Speeds up per-customer chronological queries

WHY SQLITE?
-----------
- Zero setup: ships with Python's standard library
- Persistent across restarts (file-based)
- Sufficient for a local assignment demo
- Same file used by LangGraph's SqliteSaver (checkpoint store) — later module

PATH RESOLUTION
----------------
The database path is read from .env (SQLITE_DB_PATH).
Relative paths are resolved relative to the project root (customer_support_ai/)
so the module works regardless of which directory the script is run from.
"""

import os
import sys
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

# Allow import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Resolve project root (parent of this file's directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """
    Resolve the database file path.

    Priority:
      1. Explicit db_path argument (for testing with temp DBs)
      2. SQLITE_DB_PATH environment variable
      3. Default: <project_root>/database/memory.db

    Relative paths are resolved relative to the project root so the
    database always lands in customer_support_ai/database/memory.db.

    Args:
        db_path: Optional explicit path override.

    Returns:
        Absolute path to the SQLite database file.
    """
    if db_path is not None:
        path = db_path
    else:
        path = os.getenv("SQLITE_DB_PATH", "./database/memory.db")

    # Resolve relative paths relative to project root
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)

    return path


def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open a sqlite3 connection with row_factory enabled (dict-like rows).

    Creates the database directory if it does not exist yet.

    Args:
        db_path: Optional path override (used in tests).

    Returns:
        An open sqlite3.Connection with Row factory set.
    """
    resolved = _resolve_db_path(db_path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)

    conn = sqlite3.connect(resolved)
    # Row factory lets us access columns by name: row["role"] instead of row[0]
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize the database: create the conversations table and index
    if they do not already exist.

    This function is idempotent — safe to call multiple times.
    Called once at application startup in app.py.

    Args:
        db_path: Optional path override (used in tests).

    Example:
        from memory.sqlite_memory import init_db
        init_db()  # Creates database/memory.db if not present
    """
    conn = _get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT    NOT NULL,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_customer_timestamp
                ON conversations (customer_id, timestamp);
        """)
        conn.commit()
        print(f"[Memory] Database initialized: {_resolve_db_path(db_path)}")
    finally:
        conn.close()


def save_turn(
    customer_id: str,
    session_id: str,
    role: str,
    message: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Persist one conversation turn to the database.

    Called by app.py after every graph invocation — once for the customer
    message (role="user") and once for the assistant response (role="assistant").

    Args:
        customer_id: The unique customer identifier (e.g., "CUST-001").
        session_id:  The current session ID (matches LangGraph thread_id).
        role:        Either "user" (customer message) or "assistant" (AI response).
        message:     The text content of the turn.
        db_path:     Optional path override (used in tests).

    Example:
        save_turn("CUST-001", "sess-abc", "user", "I need a refund.")
        save_turn("CUST-001", "sess-abc", "assistant", "Your refund is pending approval.")
    """
    if not message or not message.strip():
        return  # Don't save empty messages

    timestamp = datetime.now().isoformat()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO conversations
                (customer_id, session_id, role, message, timestamp)
            VALUES
                (?, ?, ?, ?, ?)
            """,
            (customer_id, session_id, role, message.strip(), timestamp),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(
    customer_id: str,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Retrieve the most recent conversation turns for a customer,
    returned in chronological order (oldest-first within the result set).

    Strategy:
      1. Fetch the latest `limit` rows ordered by timestamp DESC
      2. Reverse to return them chronologically (oldest first)

    This ensures context flows naturally into LLM prompts:
      [oldest turn] ... [most recent turn]

    Args:
        customer_id: The unique customer identifier.
        limit:       Maximum number of turns to return (default: 10).
        db_path:     Optional path override (used in tests).

    Returns:
        List of dicts, each with keys: "role", "message", "timestamp".
        Returns an empty list if no history exists.

    Example:
        history = get_history("CUST-001", limit=6)
        # [
        #   {"role": "user",      "message": "My name is David.", "timestamp": "..."},
        #   {"role": "assistant", "message": "Nice to meet you, David!", "timestamp": "..."},
        #   {"role": "user",      "message": "I have a billing issue.", "timestamp": "..."},
        # ]
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT role, message, timestamp
            FROM (
                SELECT role, message, timestamp
                FROM conversations
                WHERE customer_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) AS recent
            ORDER BY timestamp ASC
            """,
            (customer_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "role":      row["role"],
                "message":   row["message"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def clear_history(
    customer_id: str,
    db_path: Optional[str] = None,
) -> int:
    """
    Delete ALL conversation history for a specific customer.

    WARNING: This is permanent and irreversible.
    Intended for use in tests only — do not call from production code.

    Args:
        customer_id: The unique customer identifier.
        db_path:     Optional path override (used in tests).

    Returns:
        The number of rows deleted.

    Example:
        deleted = clear_history("CUST-001")
        print(f"Deleted {deleted} turns for CUST-001")
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE customer_id = ?",
            (customer_id,),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Format the conversation history list into a readable string
    suitable for injection into an LLM system prompt.

    Args:
        history: List of turn dicts from get_history().

    Returns:
        Multi-line string with each turn on its own line, prefixed
        by the role (Customer / Assistant).

    Example output:
        Customer   : My name is David.
        Assistant  : Nice to meet you, David!
        Customer   : I have a billing issue.
        Assistant  : I'll help you with that.
    """
    if not history:
        return "No previous conversation history found."

    lines = []
    for turn in history:
        role_label = "Customer  " if turn["role"] == "user" else "Assistant "
        lines.append(f"{role_label}: {turn['message']}")

    return "\n".join(lines)


def get_turn_count(customer_id: str, db_path: Optional[str] = None) -> int:
    """
    Return the total number of stored turns for a customer.
    Useful for testing and debugging.

    Args:
        customer_id: The unique customer identifier.
        db_path:     Optional path override.

    Returns:
        Integer count of all turns in the database for this customer.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE customer_id = ?",
            (customer_id,),
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()
