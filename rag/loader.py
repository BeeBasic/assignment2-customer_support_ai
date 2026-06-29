"""
rag/loader.py — PDF Document Loader and Text Splitter
======================================================

PURPOSE
-------
Loads the four TechCorp company documents from the documents/ directory,
extracts text using PyPDF, and splits them into overlapping chunks suitable
for embedding and retrieval.

DOCUMENTS LOADED
-----------------
  - company_policy.pdf    -> Policy, refund rules, cancellation terms
  - pricing_guide.pdf     -> Subscription plans, pricing, discounts
  - technical_manual.pdf  -> Troubleshooting, error codes, API reference
  - faq.pdf               -> Common questions across all domains

CHUNKING STRATEGY
------------------
chunk_size=600, chunk_overlap=100:
  - Small enough for precise semantic search (avoids diluting similarity)
  - Large enough to contain complete sentences and context
  - Overlap prevents answers from being split across chunk boundaries
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

# ── Constants ──────────────────────────────────────────────────────────────────

DOCUMENT_FILES = [
    "company_policy.pdf",
    "pricing_guide.pdf",
    "technical_manual.pdf",
    "faq.pdf",
]


def _get_docs_dir() -> str:
    """Resolve the documents directory path from .env or default."""
    docs_path = os.getenv("DOCUMENTS_PATH", "./documents")
    if not os.path.isabs(docs_path):
        docs_path = os.path.join(_PROJECT_ROOT, docs_path)
    return docs_path


# ── Public Functions ───────────────────────────────────────────────────────────

def load_documents(docs_dir: str = None) -> List[Document]:
    """
    Load all four company PDFs and return them as LangChain Document objects.

    Each Document has:
      - page_content: the text extracted from the PDF page
      - metadata:
          - source: basename of the PDF file (e.g., "pricing_guide.pdf")
          - page:   0-indexed page number within the PDF

    Args:
        docs_dir: Optional path override for the documents directory.

    Returns:
        List of LangChain Document objects from all four PDFs combined.

    Raises:
        FileNotFoundError: If a required PDF is missing.
    """
    if docs_dir is None:
        docs_dir = _get_docs_dir()

    all_documents: List[Document] = []

    for filename in DOCUMENT_FILES:
        filepath = os.path.join(docs_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required document not found: {filepath}\n"
                f"Run python create_sample_docs.py to generate sample documents."
            )

        loader = PyPDFLoader(filepath)
        pages = loader.load()

        # Normalize metadata: store only the basename as "source"
        # (The full path would make metadata bloated and non-portable)
        for page in pages:
            page.metadata["source"] = filename

        all_documents.extend(pages)
        print(f"  [Loader] Loaded {len(pages)} page(s) from {filename}")

    print(f"  [Loader] Total: {len(all_documents)} pages loaded from {len(DOCUMENT_FILES)} PDFs")
    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Split loaded Documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph
    boundaries ("\n\n") first, then sentences ("\n"), then words (" ").
    This keeps semantic units intact where possible.

    The "source" metadata field is preserved from the original Document
    so each chunk knows which PDF it came from — essential for
    collection-based routing in the vector store.

    Args:
        documents:    List of Documents from load_documents().
        chunk_size:   Maximum characters per chunk (default: 600).
        chunk_overlap: Characters of overlap between consecutive chunks (default: 100).

    Returns:
        List of smaller Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # Adds char offset metadata for debugging
    )

    chunks = splitter.split_documents(documents)
    print(f"  [Loader] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def load_and_split(docs_dir: str = None) -> List[Document]:
    """
    Convenience function: load all PDFs and split into chunks in one call.

    This is the function called by vectorstore.py during build time.

    Args:
        docs_dir: Optional path override for the documents directory.

    Returns:
        List of chunked Document objects from all four PDFs.
    """
    print("\n[RAG Loader] Loading and splitting company documents...")
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    print(f"[RAG Loader] Ready: {len(chunks)} total chunks across all documents.")
    return chunks


def get_chunks_by_source(chunks: List[Document]) -> dict:
    """
    Group chunks by their source PDF filename.

    Used by vectorstore.py to load each collection separately.

    Args:
        chunks: List of chunked Documents.

    Returns:
        Dict mapping filename -> List[Document]:
          {
            "company_policy.pdf": [...],
            "pricing_guide.pdf":  [...],
            ...
          }
    """
    grouped: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(chunk)
    return grouped
