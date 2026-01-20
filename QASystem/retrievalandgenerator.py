"""
Retrieval and Generation Pipeline
==================================
Implements the RAG (Retrieval-Augmented Generation) pipeline using Haystack AI.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from QASystem.utility import pinecone_config

# Load environment variables
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the prompt template for RAG
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information to answer the question, 
say "I don't have enough information to answer this question based on the available documents."

Context:
{% for document in documents %}
{{ document.content }}
---
{% endfor %}

Question: {{ question }}

Answer:
"""


def create_rag_pipeline(document_store, top_k: int = 3) -> Pipeline:
    """
    Create a RAG (Retrieval-Augmented Generation) pipeline.
    
    Args:
        document_store: The Pinecone document store to retrieve from.
        top_k: Number of documents to retrieve.
        
    Returns:
        A configured Haystack Pipeline for RAG.
    """
    # Initialize components
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    
    retriever = PineconeEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k
    )
    
    prompt_builder = PromptBuilder(template=RAG_PROMPT_TEMPLATE)
    
    generator = OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    # Build the pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)
    
    # Connect components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")
    
    return pipeline


# Create a global pipeline instance for reuse
_rag_pipeline: Optional[Pipeline] = None
_document_store = None


def get_rag_pipeline() -> Pipeline:
    """
    Get or create the RAG pipeline singleton.
    
    Returns:
        The RAG pipeline instance.
    """
    global _rag_pipeline, _document_store
    
    if _rag_pipeline is None:
        print("Initializing RAG pipeline...")
        _document_store = pinecone_config()
        _rag_pipeline = create_rag_pipeline(_document_store)
        
        # Warm up the embedder
        _rag_pipeline.get_component("text_embedder").warm_up()
        print("RAG pipeline initialized successfully.")
    
    return _rag_pipeline


def get_result(question: str) -> str:
    """
    Get an answer to a question using the RAG pipeline.
    
    This is the main function called by the FastAPI application.
    
    Args:
        question: The user's question.
        
    Returns:
        The generated answer based on retrieved context.
    """
    if not question or not question.strip():
        return "Please provide a valid question."
    
    try:
        # Get the pipeline
        pipeline = get_rag_pipeline()
        
        # Run the pipeline
        result = pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        
        # Extract the answer
        if result and "generator" in result:
            replies = result["generator"].get("replies", [])
            if replies:
                return replies[0]
            else:
                return "I couldn't generate an answer. Please try again."
        else:
            return "An error occurred while processing your question."
            
    except Exception as e:
        print(f"Error in get_result: {e}")
        return f"An error occurred: {str(e)}"


def retrieve_documents(question: str, top_k: int = 5) -> list:
    """
    Retrieve relevant documents without generation.
    
    Useful for debugging or displaying sources.
    
    Args:
        question: The query to search for.
        top_k: Number of documents to retrieve.
        
    Returns:
        List of retrieved documents.
    """
    document_store = pinecone_config()
    
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    text_embedder.warm_up()
    
    retriever = PineconeEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k
    )
    
    # Create a simple retrieval pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    
    result = pipeline.run({"text_embedder": {"text": question}})
    
    return result.get("retriever", {}).get("documents", [])


if __name__ == "__main__":
    # Test the RAG pipeline
    test_question = "What is this document about?"
    print(f"Question: {test_question}")
    print(f"Answer: {get_result(test_question)}")
