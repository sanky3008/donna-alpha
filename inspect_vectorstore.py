#!/usr/bin/env python3

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def inspect_vectorstore():
    """Inspect all records stored in the notes vectorstore"""
    
    print("=== Notes Vectorstore Inspection ===\n")
    
    try:
        # Initialize the same vectorstore as used by notes agent
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            collection_name="notes",
            embedding_function=embeddings,
            persist_directory="./data/chroma_db"
        )
        
        # Get the collection to access all records
        collection = vectorstore._collection
        
        print("1. Collection Information:")
        print(f"   Collection Name: {collection.name}")
        print(f"   Total Documents: {collection.count()}")
        print()
        
        # Get all documents with their metadata and IDs
        if collection.count() > 0:
            print("2. All Stored Records:")
            print("-" * 80)
            
            # Retrieve all documents (without embeddings to avoid issues)
            results = collection.get(
                include=['documents', 'metadatas']
            )
            
            for i, (doc_id, document, metadata) in enumerate(zip(
                results['ids'], 
                results['documents'], 
                results['metadatas']
            )):
                print(f"Record #{i+1}:")
                print(f"   ID: {doc_id}")
                print(f"   Content: {document}")
                print(f"   Metadata: {json.dumps(metadata, indent=6) if metadata else 'None'}")
                print("-" * 40)
        else:
            print("2. No records found in the vectorstore.")
            
        print("\n3. Test Query - Search for any notes:")
        # Try a broad search to see what comes back
        search_results = vectorstore.similarity_search("", k=10)
        print(f"   Found {len(search_results)} documents via similarity search:")
        
        for i, doc in enumerate(search_results):
            print(f"   Result #{i+1}: {doc.page_content}")
            print(f"   Metadata: {json.dumps(doc.metadata, indent=6) if doc.metadata else 'None'}")
            print()
            
    except Exception as e:
        print(f"Error inspecting vectorstore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vectorstore()