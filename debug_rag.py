#!/usr/bin/env python3
"""
Debug script to check RAG system state and test retrieval
"""

import asyncio
import sys
import os

# Add the backend app to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def debug_rag():
    """Debug the RAG system to see what's happening"""
    
    try:
        from app.services.rag_system import rag_system
        from app.core.config import settings
        
        print("🔍 Debugging RAG System...")
        print(f"📁 ChromaDB Directory: {settings.CHROMA_PERSIST_DIRECTORY}")
        print(f"🎯 Expected Embedding Dimension: {settings.BEDROCK_EMBEDDING_DIMENSION}")
        
        # Initialize the RAG system
        await rag_system.initialize()
        print("✅ RAG system initialized")
        
        # Check collections
        text_count = rag_system.text_collection.count()
        table_count = rag_system.table_collection.count()
        metadata_count = rag_system.metadata_collection.count()
        
        print(f"\n📊 Collection Counts:")
        print(f"   Text chunks: {text_count}")
        print(f"   Table chunks: {table_count}")
        print(f"   Document metadata: {metadata_count}")
        
        if metadata_count == 0:
            print("❌ No documents found in metadata collection!")
            return
        
        # Get some sample data
        print(f"\n📋 Sample Document Metadata:")
        all_metadata = rag_system.metadata_collection.get()
        for i, doc_id in enumerate(all_metadata['ids'][:3]):  # Show first 3
            metadata = all_metadata['metadatas'][i]
            print(f"   Doc {i+1}: {metadata.get('document_name', 'Unknown')} ({metadata.get('chunk_count', 0)} chunks)")
        
        # Test a simple query
        print(f"\n🔍 Testing query: 'financial summary'")
        try:
            results = await rag_system.retrieve_relevant_chunks(
                query="financial summary",
                max_results=5,
                similarity_threshold=0.1  # Very low threshold to see if anything matches
            )
            
            print(f"   Found {len(results)} results")
            for i, result in enumerate(results[:3]):
                print(f"   Result {i+1}: {result.chunk_type} (score: {result.relevance_score:.3f})")
                print(f"      Content preview: {result.content[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test embeddings
        print(f"\n🧠 Testing Embedding Generation:")
        try:
            from app.services.bedrock import bedrock_service
            await bedrock_service.initialize()
            
            test_embeddings = await bedrock_service.generate_embeddings(["test financial summary"])
            print(f"   ✅ Embedding generated, dimension: {len(test_embeddings[0])}")
            
        except Exception as e:
            print(f"   ❌ Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_rag())