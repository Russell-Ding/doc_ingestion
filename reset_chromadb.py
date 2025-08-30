#!/usr/bin/env python3
"""
Script to reset ChromaDB collections when embedding dimensions change.
Run this script when you change embedding models or dimensions.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import shutil
from pathlib import Path
import sys

def reset_chromadb():
    """Reset ChromaDB collections to use new embedding dimensions"""
    
    print("🔄 Resetting ChromaDB collections...")
    
    # Get the ChromaDB persist directory from config
    persist_dir = "./chroma_db"  # Default from config
    
    # Check if directory exists
    persist_path = Path(persist_dir)
    
    if persist_path.exists():
        print(f"📁 Found existing ChromaDB directory at: {persist_path}")
        
        # Ask for confirmation
        response = input("⚠️  This will delete all existing embeddings. Continue? (yes/no): ")
        
        if response.lower() != 'yes':
            print("❌ Cancelled. No changes made.")
            return
        
        try:
            # Remove the entire directory
            shutil.rmtree(persist_path)
            print(f"✅ Deleted ChromaDB directory: {persist_path}")
            
            # Recreate empty directory
            persist_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created new ChromaDB directory: {persist_path}")
            
        except Exception as e:
            print(f"❌ Error resetting ChromaDB: {e}")
            return
    else:
        print(f"📁 No existing ChromaDB directory found at: {persist_path}")
        persist_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created new ChromaDB directory: {persist_path}")
    
    # Initialize new collections with correct dimensions
    try:
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create the collections
        collections = [
            ("document_text", "Text content from documents"),
            ("document_tables", "Table and structured data from documents"),
            ("document_metadata", "Document metadata and summaries")
        ]
        
        for name, description in collections:
            collection = client.get_or_create_collection(
                name=name,
                metadata={"description": description}
            )
            print(f"✅ Created collection: {name}")
        
        print("\n✅ ChromaDB reset complete!")
        print("📝 Note: The collections are now ready for 1024-dimensional embeddings (Titan Embed v2)")
        print("🚀 You can now restart your backend and upload documents.")
        
    except Exception as e:
        print(f"❌ Error creating collections: {e}")
        return

if __name__ == "__main__":
    reset_chromadb()