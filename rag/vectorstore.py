"""
rag/vectorstore.py — ChromaDB Vector Store Builder
====================================================

PURPOSE
-------
Embeds the four company PDFs using nomic-embed-text (Ollama) and stores
them in four separate ChromaDB collections, one per document type.

COLLECTION ARCHITECTURE
------------------------
Using separate collections per document lets agents retrieve from only
the relevant documents instead of searching all 4 at once.

  policy_collection  <- company_policy.pdf   (all agents for policy checks)
  pricing_collection <- pricing_guide.pdf    (sales, billing agents)
  tech_collection    <- technical_manual.pdf (technical agent)
  faq_collection     <- faq.pdf              (all agents)

STORAGE
--------
ChromaDB uses a persistent client that saves to disk:
  database/chroma_db/
    chroma.sqlite3      <- ChromaDB's internal index
    [collection_uuids]/ <- One directory per collection

REBUILD BEHAVIOUR
------------------
  force_rebuild=False (default): Skips build if chroma_db directory exists
  force_rebuild=True:            Deletes existing data and rebuilds from scratch

WHY DIRECT CHROMADB?
---------------------
We use chromadb's Python client directly (not the LangChain wrapper) to
avoid deprecation warnings in langchain_community 0.3.x where Chroma
was moved to the separate `langchain_chroma` package.
"""

import os
import sys
import shutil
from typing import List, Dict, Optional

# Disable telemetry environment variable
os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Monkeypatch Posthog capture to disable telemetry printing completely
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import chromadb
from langchain_ollama import OllamaEmbeddings

from rag.loader import load_and_split, get_chunks_by_source

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

# ── Constants ──────────────────────────────────────────────────────────────────

# Maps PDF source filename -> ChromaDB collection name
DOCUMENT_COLLECTIONS: Dict[str, str] = {
    "company_policy.pdf":   "policy_collection",
    "pricing_guide.pdf":    "pricing_collection",
    "technical_manual.pdf": "tech_collection",
    "faq.pdf":              "faq_collection",
}

# All valid collection names (for existence checks)
ALL_COLLECTIONS = list(DOCUMENT_COLLECTIONS.values())


# ── Path Helpers ──────────────────────────────────────────────────────────────

def _get_chroma_dir() -> str:
    """Resolve the ChromaDB persistence directory."""
    chroma_path = os.getenv("CHROMA_DB_PATH", "./database/chroma_db")
    if not os.path.isabs(chroma_path):
        chroma_path = os.path.join(_PROJECT_ROOT, chroma_path)
    return chroma_path


def _get_embeddings() -> OllamaEmbeddings:
    """Create an OllamaEmbeddings instance for nomic-embed-text."""
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _get_client(chroma_dir: str = None) -> chromadb.PersistentClient:
    """Create or connect to the ChromaDB persistent client."""
    if chroma_dir is None:
        chroma_dir = _get_chroma_dir()
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=chroma_dir,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )


# ── Build Check ───────────────────────────────────────────────────────────────

def is_vectorstore_built(chroma_dir: str = None) -> bool:
    """
    Check whether the vector store has been built by verifying that all
    four expected collections exist in ChromaDB and have documents in them.

    Args:
        chroma_dir: Optional path override.

    Returns:
        True if all four collections exist and are non-empty.
    """
    if chroma_dir is None:
        chroma_dir = _get_chroma_dir()

    if not os.path.exists(chroma_dir):
        return False

    try:
        client = _get_client(chroma_dir)
        existing = {c.name for c in client.list_collections()}
        for col_name in ALL_COLLECTIONS:
            if col_name not in existing:
                return False
            col = client.get_collection(col_name)
            if col.count() == 0:
                return False
        return True
    except Exception:
        return False


# ── Core Build Function ───────────────────────────────────────────────────────

def build_vectorstore(force_rebuild: bool = False, chroma_dir: str = None) -> None:
    """
    Load all PDFs, embed them with nomic-embed-text, and store in ChromaDB.

    If the vectorstore already exists and force_rebuild is False, this
    function returns immediately (idempotent).

    Process:
      1. Load and split all 4 PDFs into chunks (rag/loader.py)
      2. Group chunks by source filename
      3. For each PDF -> ChromaDB collection:
           a. Get/create the collection
           b. Embed each chunk's text using OllamaEmbeddings
           c. Store (text, embedding, metadata, ID) in ChromaDB

    Args:
        force_rebuild: If True, deletes existing chroma_db and rebuilds.
        chroma_dir:    Optional path override (used in tests).
    """
    if chroma_dir is None:
        chroma_dir = _get_chroma_dir()

    # ── Skip if already built ─────────────────────────────────────────────
    if is_vectorstore_built(chroma_dir) and not force_rebuild:
        print("[VectorStore] Already built. Skipping. (Pass force_rebuild=True to rebuild)")
        return

    # ── Optional: wipe existing store ─────────────────────────────────────
    if force_rebuild and os.path.exists(chroma_dir):
        print(f"[VectorStore] force_rebuild=True — deleting {chroma_dir}")
        try:
            shutil.rmtree(chroma_dir)
        except PermissionError:
            print(f"  [VectorStore] PermissionError: directory locked. Deleting individual collections via client instead.")
            try:
                # Use a temp client to list and delete all collections
                client = _get_client(chroma_dir)
                for col in client.list_collections():
                    client.delete_collection(col.name)
            except Exception as e:
                print(f"  [VectorStore] Warning: failed to delete individual collections: {e}")

    print("\n[VectorStore] Building ChromaDB collections...")
    print(f"  Storage path: {chroma_dir}")

    # ── Load and split docs ───────────────────────────────────────────────
    all_chunks = load_and_split()
    chunks_by_source = get_chunks_by_source(all_chunks)

    # ── Create embeddings model ───────────────────────────────────────────
    embeddings = _get_embeddings()
    print(f"  [VectorStore] Embedding model: {os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}")

    # ── Create ChromaDB client ────────────────────────────────────────────
    client = _get_client(chroma_dir)

    # ── Embed and store each collection ───────────────────────────────────
    for source_file, collection_name in DOCUMENT_COLLECTIONS.items():
        chunks = chunks_by_source.get(source_file, [])
        if not chunks:
            print(f"  [VectorStore] WARNING: No chunks found for {source_file}")
            continue

        # Extract text content from Document objects
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.metadata.get("source", source_file),
                "page":   str(chunk.metadata.get("page", 0)),
            }
            for chunk in chunks
        ]

        print(f"\n  [VectorStore] Building '{collection_name}' ({len(texts)} chunks)...")

        # Compute embeddings for all chunks at once
        print(f"  [VectorStore] Embedding {len(texts)} chunks via nomic-embed-text...")
        chunk_embeddings = embeddings.embed_documents(texts)

        # Create unique IDs for each chunk
        ids = [f"{source_file}_chunk_{i}" for i in range(len(texts))]

        # Get or create the ChromaDB collection with cosine similarity
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear and re-add (handles force_rebuild and partial updates)
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
        except Exception:
            pass

        # Add all chunks to ChromaDB
        collection.add(
            documents=texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"  [VectorStore] '{collection_name}' complete: {collection.count()} vectors stored")

    print(f"\n[VectorStore] Build complete. All {len(ALL_COLLECTIONS)} collections ready.")


# ── Retrieval Interface ───────────────────────────────────────────────────────

def get_collection(collection_name: str, chroma_dir: str = None) -> chromadb.Collection:
    """
    Retrieve a single named ChromaDB collection.

    Args:
        collection_name: One of the four valid collection names.
        chroma_dir:      Optional path override.

    Returns:
        A chromadb.Collection object ready for querying.

    Raises:
        ValueError: If the collection name is not recognised.
        RuntimeError: If the vector store hasn't been built yet.
    """
    if collection_name not in ALL_COLLECTIONS:
        raise ValueError(
            f"Unknown collection '{collection_name}'. "
            f"Valid names: {ALL_COLLECTIONS}"
        )

    if chroma_dir is None:
        chroma_dir = _get_chroma_dir()

    if not is_vectorstore_built(chroma_dir):
        raise RuntimeError(
            "Vector store not built. Call build_vectorstore() first."
        )

    client = _get_client(chroma_dir)
    return client.get_collection(collection_name)


if __name__ == "__main__":
    print("Building vector store directly...")
    build_vectorstore(force_rebuild=True)
    print("Vector store build successfully completed.")

