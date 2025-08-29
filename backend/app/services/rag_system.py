import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
import structlog
from datetime import datetime
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.bedrock import bedrock_service
from app.services.document_processor import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with relevance score"""
    chunk_id: str
    document_id: str
    content: str
    chunk_type: str
    page_number: Optional[int]
    section_title: Optional[str]
    table_data: Optional[Dict[str, Any]]
    relevance_score: float
    retrieval_method: str  # 'semantic', 'keyword', 'hybrid', 'table_specific'


class RAGSystem:
    """Retrieval-Augmented Generation system optimized for mixed document types"""
    
    def __init__(self):
        self.chroma_client = None
        self.text_collection = None
        self.table_collection = None
        self.metadata_collection = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the RAG system with vector databases"""
        try:
            # Initialize ChromaDB
            if settings.CHROMA_PERSIST_DIRECTORY:
                self.chroma_client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIRECTORY,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else:
                self.chroma_client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            
            # Create collections for different content types
            self.text_collection = self.chroma_client.get_or_create_collection(
                name="document_text",
                metadata={"description": "Text content from documents"}
            )
            
            self.table_collection = self.chroma_client.get_or_create_collection(
                name="document_tables",
                metadata={"description": "Table and structured data from documents"}
            )
            
            self.metadata_collection = self.chroma_client.get_or_create_collection(
                name="document_metadata",
                metadata={"description": "Document metadata and summaries"}
            )
            
            self._initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize RAG system", error=str(e))
            raise
    
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        document_name: str
    ):
        """Add document chunks to the appropriate collections"""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Separate chunks by type
            text_chunks = []
            table_chunks = []
            
            for chunk in chunks:
                if chunk.chunk_type in ['text', 'ocr_text']:
                    text_chunks.append(chunk)
                elif chunk.chunk_type in ['table', 'excel_table', 'table_summary', 'table_row']:
                    table_chunks.append(chunk)
                else:
                    text_chunks.append(chunk)  # Default to text collection
            
            # Add text chunks
            if text_chunks:
                await self._add_text_chunks(text_chunks, document_id, document_name)
            
            # Add table chunks
            if table_chunks:
                await self._add_table_chunks(table_chunks, document_id, document_name)
            
            # Add document metadata
            await self._add_document_metadata(document_id, document_name, len(chunks))
            
            logger.info(
                "Document chunks added to RAG system",
                document_id=document_id,
                text_chunks=len(text_chunks),
                table_chunks=len(table_chunks)
            )
            
        except Exception as e:
            logger.error("Failed to add document chunks", error=str(e))
            raise
    
    async def _add_text_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        document_name: str
    ):
        """Add text chunks to the text collection"""
        
        chunk_ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embeddings for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count
            }
            
            if chunk.page_number:
                metadata["page_number"] = chunk.page_number
            
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            
            metadatas.append(metadata)
        
        self.text_collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def _add_table_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        document_name: str
    ):
        """Add table chunks to the table collection"""
        
        chunk_ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embeddings for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count
            }
            
            if chunk.page_number:
                metadata["page_number"] = chunk.page_number
            
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            
            if chunk.table_data:
                # Store essential table metadata
                metadata["table_row_count"] = chunk.table_data.get("row_count", 0)
                metadata["table_column_count"] = chunk.table_data.get("column_count", 0)
                metadata["table_headers"] = json.dumps(chunk.table_data.get("headers", []))
                # Store full table data separately (ChromaDB has metadata size limits)
                metadata["has_table_data"] = True
            
            metadatas.append(metadata)
        
        self.table_collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def _add_document_metadata(
        self,
        document_id: str,
        document_name: str,
        chunk_count: int
    ):
        """Add document-level metadata"""
        
        # Generate embedding for document name and summary
        summary_text = f"Document: {document_name}, Total chunks: {chunk_count}"
        embeddings = await bedrock_service.generate_embeddings([summary_text])
        
        self.metadata_collection.add(
            ids=[document_id],
            embeddings=embeddings,
            documents=[summary_text],
            metadatas=[{
                "document_id": document_id,
                "document_name": document_name,
                "chunk_count": chunk_count,
                "added_date": datetime.now().isoformat()
            }]
        )
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_results: int = None,
        similarity_threshold: float = None,
        include_tables: bool = True,
        focus_keywords: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks using hybrid search"""
        
        if not self._initialized:
            await self.initialize()
        
        max_results = max_results or settings.MAX_RETRIEVED_CHUNKS
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        try:
            # Generate query embedding
            query_embeddings = await bedrock_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            results = []
            
            # Semantic search in text collection
            text_results = await self._semantic_search(
                query_embedding,
                self.text_collection,
                max_results=max_results // 2,
                document_types=document_types,
                retrieval_method="semantic"
            )
            results.extend(text_results)
            
            # Semantic search in table collection (if enabled)
            if include_tables:
                table_results = await self._semantic_search(
                    query_embedding,
                    self.table_collection,
                    max_results=max_results // 2,
                    document_types=document_types,
                    retrieval_method="semantic"
                )
                results.extend(table_results)
            
            # Keyword-based search for specific terms
            if focus_keywords:
                keyword_results = await self._keyword_search(
                    focus_keywords,
                    document_types=document_types,
                    max_results=max_results // 4
                )
                results.extend(keyword_results)
            
            # Table-specific search for numerical/financial queries
            if include_tables and self._is_numerical_query(query):
                numerical_results = await self._numerical_table_search(
                    query,
                    document_types=document_types,
                    max_results=max_results // 4
                )
                results.extend(numerical_results)
            
            # Deduplicate and rank results
            final_results = self._deduplicate_and_rank_results(
                results,
                similarity_threshold,
                max_results
            )
            
            logger.info(
                "Retrieved relevant chunks",
                query_length=len(query),
                total_results=len(final_results),
                text_results=len([r for r in final_results if r.chunk_type in ['text', 'ocr_text']]),
                table_results=len([r for r in final_results if 'table' in r.chunk_type])
            )
            
            return final_results
            
        except Exception as e:
            logger.error("Chunk retrieval failed", error=str(e))
            raise
    
    async def _semantic_search(
        self,
        query_embedding: List[float],
        collection,
        max_results: int,
        document_types: Optional[List[str]] = None,
        retrieval_method: str = "semantic"
    ) -> List[RetrievalResult]:
        """Perform semantic search on a collection"""
        
        where_clause = {}
        if document_types:
            where_clause["chunk_type"] = {"$in": document_types}
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause if where_clause else None
            )
            
            retrieval_results = []
            for i, chunk_id in enumerate(results['ids'][0]):
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=results['metadatas'][0][i]['document_id'],
                    content=results['documents'][0][i],
                    chunk_type=results['metadatas'][0][i]['chunk_type'],
                    page_number=results['metadatas'][0][i].get('page_number'),
                    section_title=results['metadatas'][0][i].get('section_title'),
                    table_data=None,  # Will be loaded separately if needed
                    relevance_score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    retrieval_method=retrieval_method
                )
                retrieval_results.append(result)
            
            return retrieval_results
            
        except Exception as e:
            logger.warning("Semantic search failed", error=str(e))
            return []
    
    async def _keyword_search(
        self,
        keywords: List[str],
        document_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[RetrievalResult]:
        """Perform keyword-based search"""
        
        results = []
        
        try:
            # Search in text collection
            for keyword in keywords:
                where_clause = {"$contains": keyword.lower()}
                if document_types:
                    where_clause = {"$and": [where_clause, {"chunk_type": {"$in": document_types}}]}
                
                text_results = self.text_collection.get(
                    where_document=where_clause,
                    limit=max_results // len(keywords)
                )
                
                for i, chunk_id in enumerate(text_results['ids']):
                    result = RetrievalResult(
                        chunk_id=chunk_id,
                        document_id=text_results['metadatas'][i]['document_id'],
                        content=text_results['documents'][i],
                        chunk_type=text_results['metadatas'][i]['chunk_type'],
                        page_number=text_results['metadatas'][i].get('page_number'),
                        section_title=text_results['metadatas'][i].get('section_title'),
                        table_data=None,
                        relevance_score=0.8,  # Fixed score for keyword matches
                        retrieval_method="keyword"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning("Keyword search failed", error=str(e))
            return []
    
    async def _numerical_table_search(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """Search for tables with numerical data relevant to the query"""
        
        try:
            # Look for tables with numerical content
            where_clause = {
                "$and": [
                    {"chunk_type": {"$in": ["table", "excel_table", "table_summary"]}},
                    {"table_row_count": {"$gt": 0}}
                ]
            }
            
            if document_types:
                where_clause["$and"].append({"chunk_type": {"$in": document_types}})
            
            table_results = self.table_collection.get(
                where=where_clause,
                limit=max_results
            )
            
            results = []
            for i, chunk_id in enumerate(table_results['ids']):
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=table_results['metadatas'][i]['document_id'],
                    content=table_results['documents'][i],
                    chunk_type=table_results['metadatas'][i]['chunk_type'],
                    page_number=table_results['metadatas'][i].get('page_number'),
                    section_title=table_results['metadatas'][i].get('section_title'),
                    table_data=None,  # Will be loaded from database if needed
                    relevance_score=0.7,  # Score for numerical relevance
                    retrieval_method="table_specific"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning("Numerical table search failed", error=str(e))
            return []
    
    def _is_numerical_query(self, query: str) -> bool:
        """Check if query is asking for numerical/financial information"""
        numerical_keywords = [
            'financial', 'revenue', 'profit', 'loss', 'amount', 'total', 'sum',
            'percentage', 'ratio', 'rate', 'cost', 'expense', 'income', 'value',
            'price', 'budget', 'forecast', 'growth', 'decline', 'metrics',
            'performance', 'statistics', 'data', 'numbers'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in numerical_keywords)
    
    def _deduplicate_and_rank_results(
        self,
        results: List[RetrievalResult],
        similarity_threshold: float,
        max_results: int
    ) -> List[RetrievalResult]:
        """Remove duplicates and rank results by relevance"""
        
        # Remove duplicates by chunk_id
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in unique_results
            if result.relevance_score >= similarity_threshold
        ]
        
        # Sort by relevance score (descending)
        ranked_results = sorted(
            filtered_results,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        # Return top results
        return ranked_results[:max_results]
    
    async def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a document"""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            metadata_result = self.metadata_collection.get(
                ids=[document_id]
            )
            
            if metadata_result['ids']:
                metadata = metadata_result['metadatas'][0]
                
                # Get chunk statistics
                text_chunks = self.text_collection.get(
                    where={"document_id": document_id}
                )
                
                table_chunks = self.table_collection.get(
                    where={"document_id": document_id}
                )
                
                return {
                    "document_id": document_id,
                    "document_name": metadata["document_name"],
                    "total_chunks": metadata["chunk_count"],
                    "text_chunks": len(text_chunks['ids']),
                    "table_chunks": len(table_chunks['ids']),
                    "added_date": metadata["added_date"]
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get document summary", error=str(e))
            return None
    
    async def delete_document(self, document_id: str):
        """Remove all chunks for a document from the RAG system"""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all chunks for this document
            text_chunks = self.text_collection.get(
                where={"document_id": document_id}
            )
            
            table_chunks = self.table_collection.get(
                where={"document_id": document_id}
            )
            
            # Delete from collections
            if text_chunks['ids']:
                self.text_collection.delete(ids=text_chunks['ids'])
            
            if table_chunks['ids']:
                self.table_collection.delete(ids=table_chunks['ids'])
            
            # Delete metadata
            self.metadata_collection.delete(ids=[document_id])
            
            logger.info(
                "Document deleted from RAG system",
                document_id=document_id,
                text_chunks_deleted=len(text_chunks['ids']),
                table_chunks_deleted=len(table_chunks['ids'])
            )
            
        except Exception as e:
            logger.error("Failed to delete document", error=str(e))
            raise


# Global RAG system instance
rag_system = RAGSystem()