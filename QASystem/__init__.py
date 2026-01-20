"""
QASystem Package
================
End-to-End RAG (Retrieval-Augmented Generation) System using Haystack AI and Pinecone.

This package provides:
- Document ingestion and indexing
- Retrieval and generation pipeline
- Utility functions for configuration
"""

from QASystem.utility import pinecone_config
from QASystem.retrievalandgenerator import get_result
from QASystem.ingestion import ingest_documents

__all__ = [
    "pinecone_config",
    "get_result", 
    "ingest_documents"
]
