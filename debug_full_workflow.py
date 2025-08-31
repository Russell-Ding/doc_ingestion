#!/usr/bin/env python3
"""
Complete debug script that mimics the full RAG generation workflow
Tests: Document Upload -> RAG Retrieval -> Content Generation
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add the backend app to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_full_workflow():
    """Test the complete workflow from document retrieval to content generation"""
    
    print("🚀 Testing Full RAG Workflow")
    print("=" * 50)
    
    try:
        # Import services
        from app.services.rag_system import rag_system
        from app.services.bedrock import bedrock_service
        from app.services.agents import document_finder_agent, content_generator_agent
        from app.core.config import settings
        
        # Step 1: Check RAG System State
        print("\n1️⃣ CHECKING RAG SYSTEM STATE")
        print("-" * 30)
        
        await rag_system.initialize()
        print("✅ RAG system initialized")
        
        # Check collections
        text_count = rag_system.text_collection.count()
        table_count = rag_system.table_collection.count()
        metadata_count = rag_system.metadata_collection.count()
        
        print(f"📊 Collection Counts:")
        print(f"   Text chunks: {text_count}")
        print(f"   Table chunks: {table_count}")
        print(f"   Document metadata: {metadata_count}")
        
        if metadata_count == 0:
            print("❌ ERROR: No documents found! Upload some documents first.")
            return
        
        # Show document list
        print(f"\n📋 Available Documents:")
        all_metadata = rag_system.metadata_collection.get()
        for i, doc_id in enumerate(all_metadata['ids']):
            metadata = all_metadata['metadatas'][i]
            print(f"   {i+1}. {metadata.get('document_name', 'Unknown')} "
                  f"({metadata.get('chunk_count', 0)} chunks, "
                  f"uploaded: {metadata.get('added_date', 'Unknown')})")
        
        # Step 2: Test Embedding Generation
        print("\n2️⃣ TESTING EMBEDDING GENERATION")
        print("-" * 35)
        
        await bedrock_service.initialize()
        print("✅ Bedrock service initialized")
        
        test_query = "financial summary analysis revenue cash flow"
        print(f"🧠 Generating embedding for: '{test_query}'")
        
        query_embeddings = await bedrock_service.generate_embeddings([test_query])
        query_embedding = query_embeddings[0]
        print(f"✅ Embedding generated, dimension: {len(query_embedding)}")
        print(f"   First 5 values: {query_embedding[:5]}")
        
        # Step 3: Test Direct RAG Retrieval
        print("\n3️⃣ TESTING RAG RETRIEVAL")
        print("-" * 25)
        
        # Test with very low threshold to see if anything matches
        print("🔍 Testing with low similarity threshold (0.0)...")
        results_low = await rag_system.retrieve_relevant_chunks(
            query=test_query,
            max_results=10,
            similarity_threshold=0.0,  # Accept anything
            include_tables=True
        )
        
        print(f"   Found {len(results_low)} results with threshold 0.0")
        if results_low:
            for i, result in enumerate(results_low[:3]):
                print(f"   Result {i+1}: {result.chunk_type} "
                      f"(score: {result.relevance_score:.3f}, "
                      f"doc: {result.document_id[:8]}...)")
                print(f"      Preview: {result.content[:150]}...")
        
        # Test with normal threshold
        print(f"\n🎯 Testing with normal threshold ({settings.SIMILARITY_THRESHOLD})...")
        results_normal = await rag_system.retrieve_relevant_chunks(
            query=test_query,
            max_results=10,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            include_tables=True
        )
        
        print(f"   Found {len(results_normal)} results with normal threshold")
        
        # Step 4: Test Document Finder Agent
        print("\n4️⃣ TESTING DOCUMENT FINDER AGENT")
        print("-" * 30)
        
        finder_task = {
            "segment_prompt": "Generate a comprehensive financial summary including revenue analysis, cash flow trends, and key financial ratios",
            "required_document_types": [],
            "max_documents": 5
        }
        
        print(f"🤖 Running DocumentFinderAgent...")
        print(f"   Prompt: {finder_task['segment_prompt'][:100]}...")
        
        finder_result = await document_finder_agent.execute(finder_task)
        
        if finder_result.success:
            print(f"✅ Document finder succeeded")
            docs_found = finder_result.data.get("documents_found", [])
            print(f"   Documents found: {len(docs_found)}")
            print(f"   Total chunks: {finder_result.data.get('total_chunks', 0)}")
            
            for i, doc in enumerate(docs_found[:3]):
                print(f"   Doc {i+1}: {doc['document_id'][:8]}... "
                      f"({len(doc.get('chunks', []))} chunks, "
                      f"relevance: {doc.get('total_relevance_score', 0):.3f})")
        else:
            print(f"❌ Document finder failed: {finder_result.error_message}")
            return
        
        # Step 5: Test Content Generator Agent
        print("\n5️⃣ TESTING CONTENT GENERATOR AGENT")
        print("-" * 32)
        
        if docs_found:
            generator_task = {
                "segment_prompt": "Generate a comprehensive financial summary including revenue analysis, cash flow trends, and key financial ratios",
                "segment_name": "Financial Summary",
                "retrieved_documents": docs_found,
                "generation_settings": {}
            }
            
            print(f"🤖 Running ContentGeneratorAgent...")
            print(f"   Using {len(docs_found)} retrieved documents")
            
            generator_result = await content_generator_agent.execute(generator_task)
            
            if generator_result.success:
                print(f"✅ Content generator succeeded")
                content = generator_result.data.get("generated_content", "")
                print(f"   Generated content length: {len(content)} characters")
                print(f"   Content preview: {content[:200]}...")
                print(f"   References: {len(generator_result.data.get('references', []))}")
            else:
                print(f"❌ Content generator failed: {generator_result.error_message}")
        else:
            print("⚠️ Skipping content generation - no documents found")
        
        # Step 6: Direct Collection Query Test
        print("\n6️⃣ TESTING DIRECT COLLECTION QUERIES")
        print("-" * 35)
        
        # Test direct text collection query
        print("🔍 Direct text collection query...")
        try:
            direct_text = rag_system.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            print(f"   Direct text query returned {len(direct_text.get('ids', [[]]))} results")
            if direct_text.get('ids') and direct_text['ids'][0]:
                print(f"   Sample distance: {direct_text.get('distances', [[]])[0][0] if direct_text.get('distances') else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Direct text query failed: {e}")
        
        # Test direct table collection query
        print("🔍 Direct table collection query...")
        try:
            direct_table = rag_system.table_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            print(f"   Direct table query returned {len(direct_table.get('ids', [[]]))} results")
        except Exception as e:
            print(f"   ❌ Direct table query failed: {e}")
        
        # Step 7: Check Configuration
        print("\n7️⃣ CONFIGURATION CHECK")
        print("-" * 20)
        
        print(f"📋 Key Settings:")
        print(f"   MAX_RETRIEVED_CHUNKS: {settings.MAX_RETRIEVED_CHUNKS}")
        print(f"   SIMILARITY_THRESHOLD: {settings.SIMILARITY_THRESHOLD}")
        print(f"   BEDROCK_EMBEDDING_MODEL: {settings.BEDROCK_EMBEDDING_MODEL}")
        print(f"   BEDROCK_EMBEDDING_DIMENSION: {settings.BEDROCK_EMBEDDING_DIMENSION}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 RAG System Debug Tool")
    print("This script tests the complete document retrieval and generation workflow")
    print()
    
    asyncio.run(test_full_workflow())