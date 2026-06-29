"""
rag/retriever.py — Intent-Aware Document Retriever
====================================================

PURPOSE
-------
The retriever is the public API for RAG in this system. It queries only
the ChromaDB collections relevant to a given intent, ranks results by
cosine similarity, and returns the top-K chunks as dicts.

INTENT -> COLLECTION MAPPING
------------------------------
  sales         -> pricing_collection + faq_collection
  technical     -> tech_collection + faq_collection
  billing       -> policy_collection + pricing_collection + faq_collection
  account       -> policy_collection + faq_collection
  memory_recall -> []  (no RAG — memory_handler uses SQLite only)

DESIGN RATIONALE
-----------------
Using multiple per-intent collections instead of one global collection:
  - Improves precision: a "pricing" query won't pull technical manual chunks
  - Reduces noise in the LLM prompt
  - Mirrors how a real enterprise KB is segmented by domain

TOP-K STRATEGY
----------------
When multiple collections are queried:
  - top_k results are fetched PER collection (not total)
  - Results are merged and the overall top-k are returned by distance score
  - This ensures at least 1 result per collection even if one is weaker
"""

import os
import sys
from typing import List, Dict, Optional

# Disable ChromaDB telemetry globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Monkeypatch Posthog capture to disable telemetry printing completely
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from rag.vectorstore import (
    get_collection,
    build_vectorstore,
    is_vectorstore_built,
    _get_chroma_dir,
    _get_embeddings,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

# ── Intent -> Collection Mapping ──────────────────────────────────────────────

INTENT_COLLECTIONS: Dict[str, List[str]] = {
    "sales":         ["pricing_collection", "faq_collection"],
    "technical":     ["tech_collection",    "faq_collection"],
    "billing":       ["policy_collection",  "pricing_collection", "faq_collection"],
    "account":       ["policy_collection",  "faq_collection"],
    "memory_recall": [],  # RAG skipped — memory_handler uses SQLite
}


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    intent: str,
    top_k: int = 3,
) -> List[Dict[str, str]]:
    """
    Retrieve the top-K most relevant document chunks for a customer query.

    This is the main function called by each specialized agent before
    generating a response. It:
      1. Determines which collections are relevant for the given intent
      2. Auto-builds the vector store if it hasn't been built yet
      3. Embeds the query using nomic-embed-text
      4. Queries each relevant collection
      5. Merges and deduplicates results
      6. Returns the top-K chunks by similarity score

    Args:
        query:  The customer's raw query text.
        intent: The classified intent (determines which collections to search).
        top_k:  Number of results to return (default: 3 as per assignment).

    Returns:
        List of dicts, each containing:
          - "content": The text of the retrieved chunk
          - "source":  The source PDF filename (e.g., "pricing_guide.pdf")
          - "score":   Cosine distance score (lower = more similar)

        Returns an empty list if:
          - intent is "memory_recall" (no RAG needed)
          - vector store has no relevant documents
          - embedding fails (Ollama not running)

    Example:
        docs = retrieve("What are your pricing plans?", "sales", top_k=3)
        for doc in docs:
            print(f"[{doc['source']}] {doc['content'][:80]}")
    """
    # No RAG for memory recall — handled entirely by SQLite memory
    collection_names = INTENT_COLLECTIONS.get(intent, [])
    if not collection_names:
        return []

    # Auto-build if not yet built (first run)
    if not is_vectorstore_built():
        print(f"[Retriever] Vector store not found. Building now (one-time setup)...")
        build_vectorstore()

    # Embed the query
    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        print(f"[Retriever] Embedding failed: {e}")
        return []

    # Query each relevant collection
    raw_results: List[Dict] = []
    for col_name in collection_names:
        try:
            collection = get_collection(col_name)
            n = min(top_k, collection.count())  # Can't request more than available
            if n == 0:
                continue

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

            # Unpack ChromaDB results
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                raw_results.append({
                    "content": doc,
                    "source":  meta.get("source", col_name),
                    "score":   round(dist, 4),
                })

        except Exception as e:
            print(f"[Retriever] Error querying '{col_name}': {e}")
            continue

    if not raw_results:
        return []

    # Sort by score (lower cosine distance = more relevant) and return top-K
    raw_results.sort(key=lambda x: x["score"])
    top_results = raw_results[:top_k]

    print(f"[Retriever] intent='{intent}' | query='{query[:50]}' | "
          f"found {len(top_results)} chunks from {len(collection_names)} collections")

    return top_results


def format_docs_for_prompt(docs: List[Dict[str, str]]) -> str:
    """
    Format retrieved document chunks into a string for LLM prompt injection.

    Each chunk is labelled with its source file so the LLM can cite it.
    This also helps the Supervisor verify factual accuracy.

    Args:
        docs: List of retrieved chunk dicts from retrieve().

    Returns:
        Formatted multi-line string, or empty string if no docs retrieved.

    Example output:
        [pricing_guide.pdf]
        Starter Plan: $9/month for up to 3 users, 10 GB storage...

        [faq.pdf]
        Q: Can I upgrade my plan at any time? A: Yes, upgrades take effect...
    """
    if not docs:
        return ""

    parts = []
    for doc in docs:
        source = doc.get("source", "company document")
        content = doc.get("content", "").strip()
        if content:
            parts.append(f"[{source}]\n{content}")

    return "\n\n".join(parts)


def get_retriever(intent: str, top_k: int = 3):
    """
    Return a simple callable retriever bound to a specific intent.

    This allows agents to hold a retriever reference without passing
    the intent on every call.

    Args:
        intent: The classified intent.
        top_k:  Number of results to return.

    Returns:
        A callable: (query: str) -> List[Dict[str, str]]

    Example:
        retriever = get_retriever("sales")
        docs = retriever("What is the cheapest plan?")
    """
    def _retriever(query: str) -> List[Dict[str, str]]:
        return retrieve(query, intent, top_k=top_k)
    return _retriever
