"""Transparent, small shared utilities for the hands-on RAG notebooks."""
from .core import (
    Document, Settings, settings, offline_mode, get_client, anthropic_headers, load_documents,
    chunk_documents, TfidfIndex, DenseIndex, bm25_search,
    reciprocal_rank_fusion, context_text, ask_claude, rag_answer, show_hits,
    retrieval_metrics, GROUNDED_SYSTEM_PROMPT,
)
