"""
Document Ingestion Pipeline
============================
Handles document loading, preprocessing, embedding generation, and storage in Pinecone.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from haystack import Pipeline, Document
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from QASystem.utility import pinecone_config

# Load environment variables
load_dotenv()


def create_ingestion_pipeline(document_store) -> Pipeline:
    """
    Create a Haystack pipeline for document ingestion.
    
    Args:
        document_store: The Pinecone document store to write documents to.
        
    Returns:
        A configured Haystack Pipeline for document ingestion.
    """
    # Initialize components
    document_cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False
    )
    
    document_splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1
    )
    
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    
    document_writer = DocumentWriter(
        document_store=document_store,
        policy="upsert"
    )
    
    # Build the pipeline
    pipeline = Pipeline()
    pipeline.add_component("cleaner", document_cleaner)
    pipeline.add_component("splitter", document_splitter)
    pipeline.add_component("embedder", document_embedder)
    pipeline.add_component("writer", document_writer)
    
    # Connect components
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    
    return pipeline


def load_documents_from_directory(directory_path: str, file_types: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from a directory.
    
    Args:
        directory_path: Path to the directory containing documents.
        file_types: List of file extensions to process (e.g., ['.txt', '.pdf']).
                   Defaults to ['.txt', '.pdf'].
                   
    Returns:
        List of Haystack Document objects.
    """
    if file_types is None:
        file_types = ['.txt', '.pdf']
    
    documents = []
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in file_types:
            try:
                if file_path.suffix.lower() == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc = Document(
                        content=content,
                        meta={"source": str(file_path), "filename": file_path.name}
                    )
                    documents.append(doc)
                    print(f"Loaded: {file_path.name}")
                elif file_path.suffix.lower() == '.pdf':
                    # For PDF files, we'll use PyPDFToDocument converter
                    converter = PyPDFToDocument()
                    result = converter.run(sources=[file_path])
                    for doc in result["documents"]:
                        doc.meta["source"] = str(file_path)
                        doc.meta["filename"] = file_path.name
                        documents.append(doc)
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    
    return documents


def ingest_documents(source_path: str, file_types: Optional[List[str]] = None) -> dict:
    """
    Main function to ingest documents into Pinecone.
    
    Args:
        source_path: Path to directory containing documents or a single file.
        file_types: List of file extensions to process.
        
    Returns:
        Dictionary with ingestion results.
    """
    print("Starting document ingestion...")
    
    # Initialize document store
    document_store = pinecone_config()
    print("Pinecone document store initialized.")
    
    # Load documents
    source = Path(source_path)
    
    if source.is_file():
        # Single file ingestion
        if source.suffix.lower() == '.txt':
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(
                content=content,
                meta={"source": str(source), "filename": source.name}
            )]
        elif source.suffix.lower() == '.pdf':
            converter = PyPDFToDocument()
            result = converter.run(sources=[source])
            documents = result["documents"]
            for doc in documents:
                doc.meta["source"] = str(source)
                doc.meta["filename"] = source.name
        else:
            raise ValueError(f"Unsupported file type: {source.suffix}")
    elif source.is_dir():
        documents = load_documents_from_directory(str(source), file_types)
    else:
        raise FileNotFoundError(f"Source path not found: {source_path}")
    
    if not documents:
        return {"status": "warning", "message": "No documents found to ingest.", "count": 0}
    
    print(f"Loaded {len(documents)} documents.")
    
    # Create and run ingestion pipeline
    pipeline = create_ingestion_pipeline(document_store)
    
    # Warm up embedder
    pipeline.get_component("embedder").warm_up()
    
    # Run the pipeline
    result = pipeline.run({"cleaner": {"documents": documents}})
    
    written_count = result.get("writer", {}).get("documents_written", len(documents))
    
    print(f"Successfully ingested {written_count} document chunks into Pinecone.")
    
    return {
        "status": "success",
        "message": f"Ingested {written_count} document chunks.",
        "count": written_count
    }


if __name__ == "__main__":
    # Example usage - ingest documents from a 'data' directory
    import sys
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = "./data"
        
    try:
        result = ingest_documents(source)
        print(result)
    except Exception as e:
        print(f"Ingestion failed: {e}")
