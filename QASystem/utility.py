"""
Utility Functions for QASystem
==============================
Configuration and helper functions for Pinecone and other services.
"""

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    print("Pinecone API key loaded successfully.")
else:
    print("Warning: PINECONE_API_KEY not found in environment variables.")


def pinecone_config(
    index_name: str = "quickstart",
    namespace: str = "default",
    dimension: int = 768,
    metric: str = "cosine"
) -> PineconeDocumentStore:
    """
    Configure and return a Pinecone document store.
    
    Args:
        index_name: Name of the Pinecone index (default: "quickstart")
        namespace: Namespace within the index (default: "default")
        dimension: Embedding dimension, should match your embedder model
                   (768 for all-mpnet-base-v2, 384 for all-MiniLM-L6-v2)
        metric: Distance metric for similarity search (default: "cosine")
        
    Returns:
        Configured PineconeDocumentStore instance.
        
    Note:
        Make sure to set PINECONE_API_KEY in your environment variables or .env file.
        You can get your API key from https://www.pinecone.io/
    """
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError(
            "PINECONE_API_KEY not found. Please set it in your .env file or environment variables."
        )
    
    document_store = PineconeDocumentStore(
        index=index_name,
        namespace=namespace,
        dimension=dimension,
        metric=metric
    )
    
    return document_store


def get_environment_info() -> dict:
    """
    Get information about the current environment configuration.
    
    Returns:
        Dictionary with environment information.
    """
    return {
        "pinecone_api_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
        "environment": os.getenv("ENVIRONMENT", "development")
    }


if __name__ == "__main__":
    # Test configuration
    print("Environment Info:", get_environment_info())
    
    try:
        store = pinecone_config()
        print("Pinecone document store configured successfully!")
    except Exception as e:
        print(f"Error configuring Pinecone: {e}")