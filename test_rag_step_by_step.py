#!/usr/bin/env python3
"""
Step-by-step RAG system test script
Run this to debug why documents aren't being retrieved
"""

import asyncio
import sys
import os

# Add the backend app to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def main():
    print("🔍 RAG System Step-by-Step Test")
    print("=" * 40)
    
    # Step 1: Basic imports and initialization
    print("\n📦 Step 1: Importing services...")
    try:
        from app.services.rag_system import rag_system
        from app.services.bedrock import bedrock_service
        from app.core.config import settings
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Step 2: Initialize services
    print("\n🚀 Step 2: Initializing services...")
    try:
        await rag_system.initialize()
        await bedrock_service.initialize()
        print("✅ Services initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Step 3: Check document count
    print("\n📊 Step 3: Checking document collections...")
    try:
        text_count = rag_system.text_collection.count()
        table_count = rag_system.table_collection.count()
        metadata_count = rag_system.metadata_collection.count()
        
        print(f"   Text chunks: {text_count}")
        print(f"   Table chunks: {table_count}")
        print(f"   Document metadata: {metadata_count}")
        
        if metadata_count == 0:
            print("❌ No documents found! Documents may not have been stored properly.")
            return
        
        # Show document names
        docs = rag_system.metadata_collection.get()
        print(f"\n📋 Documents in system:")
        for i, doc_id in enumerate(docs['ids']):
            meta = docs['metadatas'][i]
            print(f"   {i+1}. {meta.get('document_name', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Collection check failed: {e}")
        return
    
    # Step 4: Test embedding generation
    print("\n🧠 Step 4: Testing embedding generation...")
    test_query = "financial summary"
    try:
        embeddings = await bedrock_service.generate_embeddings([test_query])
        print(f"✅ Embedding generated: {len(embeddings[0])} dimensions")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return
    
    # Step 5: Test direct ChromaDB query
    print("\n🔍 Step 5: Testing direct ChromaDB query...")
    try:
        # Query with very low threshold
        direct_results = rag_system.text_collection.query(
            query_embeddings=[embeddings[0]],
            n_results=5
        )
        
        if direct_results and direct_results.get('ids') and direct_results['ids'][0]:
            print(f"✅ Direct query found {len(direct_results['ids'][0])} results")
            print(f"   Best distance: {min(direct_results['distances'][0]):.4f}")
            print(f"   Worst distance: {max(direct_results['distances'][0]):.4f}")
        else:
            print("❌ Direct query returned no results")
            
    except Exception as e:
        print(f"❌ Direct query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Test RAG retrieve_relevant_chunks
    print("\n🎯 Step 6: Testing RAG retrieve_relevant_chunks...")
    try:
        results = await rag_system.retrieve_relevant_chunks(
            query=test_query,
            max_results=5,
            similarity_threshold=0.0  # Accept everything
        )
        
        print(f"   RAG retrieval found: {len(results)} results")
        if results:
            for i, result in enumerate(results):
                print(f"   {i+1}. {result.chunk_type} (score: {result.relevance_score:.4f})")
        else:
            print("❌ RAG retrieval returned no results")
            
    except Exception as e:
        print(f"❌ RAG retrieval failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Test with different queries
    print("\n🔄 Step 7: Testing different query types...")
    test_queries = [
        "revenue",
        "cash flow", 
        "financial data",
        "income statement",
        "balance sheet"
    ]
    
    for query in test_queries:
        try:
            results = await rag_system.retrieve_relevant_chunks(
                query=query,
                max_results=3,
                similarity_threshold=0.0
            )
            print(f"   '{query}': {len(results)} results")
        except Exception as e:
            print(f"   '{query}': ERROR - {e}")
    
    print("\n✅ Debug complete!")
    print("\nNext steps:")
    print("1. If collections are empty: Check document upload process")
    print("2. If embeddings fail: Check bedrock_utils.py get_runtime function")
    print("3. If queries return no results: Check similarity threshold settings")
    print("4. If direct queries work but RAG doesn't: Check RAG filtering logic")

if __name__ == "__main__":
    asyncio.run(main())