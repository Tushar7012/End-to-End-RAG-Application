from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
print("Import Successfully")

def pinecone_config():
    #configuring pinecone database
    document_store = PineconeDocumentStore(
            environment="gcp-starter",
            index="default",
            namespace="default",
            dimension=768
        )
    return document_store