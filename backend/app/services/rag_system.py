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
            
            # Check and recreate collections if dimension mismatch
            await self._ensure_correct_collections()
            
            self._initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize RAG system", error=str(e))
            raise
    
    async def _ensure_correct_collections(self):
        """Ensure collections exist with correct embedding dimensions"""
        collections_to_create = [
            ("document_text", "Text content from documents"),
            ("document_tables", "Table and structured data from documents"),
            ("document_metadata", "Document metadata and summaries")
        ]
        
        for collection_name, description in collections_to_create:
            try:
                # Try to get existing collection
                collection = self.chroma_client.get_collection(collection_name)
                
                # Check if collection has data and verify dimensions
                if collection.count() > 0:
                    # Try to peek at the data to check dimensions
                    try:
                        result = collection.peek(1)
                        if result and 'embeddings' in result and result['embeddings']:
                            existing_dim = len(result['embeddings'][0])
                            expected_dim = settings.BEDROCK_EMBEDDING_DIMENSION
                            
                            if existing_dim != expected_dim:
                                logger.warning(
                                    f"Collection {collection_name} has wrong dimensions",
                                    existing=existing_dim,
                                    expected=expected_dim
                                )
                                # Delete and recreate collection
                                self.chroma_client.delete_collection(collection_name)
                                logger.info(f"Deleted collection {collection_name} due to dimension mismatch")
                                collection = self.chroma_client.create_collection(
                                    name=collection_name,
                                    metadata={"description": description, "dimension": expected_dim}
                                )
                                logger.info(f"Recreated collection {collection_name} with {expected_dim} dimensions")
                    except Exception as e:
                        logger.warning(f"Could not check dimensions for {collection_name}: {e}")
                
            except Exception:
                # Collection doesn't exist, create it
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": description, "dimension": settings.BEDROCK_EMBEDDING_DIMENSION}
                )
                logger.info(f"Created new collection {collection_name}")
            
            # Assign to appropriate attribute
            if collection_name == "document_text":
                self.text_collection = collection
            elif collection_name == "document_tables":
                self.table_collection = collection
            elif collection_name == "document_metadata":
                self.metadata_collection = collection
    
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        document_name: str,
        file_size: int = 0
    ):
        """Add document chunks to the appropriate collections"""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate embedding dimensions
            if chunks and chunks[0].embeddings:
                actual_dim = len(chunks[0].embeddings)
                expected_dim = settings.BEDROCK_EMBEDDING_DIMENSION
                
                if actual_dim != expected_dim:
                    logger.error(
                        "Embedding dimension mismatch",
                        actual=actual_dim,
                        expected=expected_dim
                    )
                    raise ValueError(
                        f"Embedding dimension mismatch: got {actual_dim}, "
                        f"expected {expected_dim}. Run reset_chromadb.py to fix."
                    )
            
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
            await self._add_document_metadata(document_id, document_name, len(chunks), file_size)
            
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
                metadata["page_number"] = int(chunk.page_number) if chunk.page_number else None
            
            if chunk.section_title:
                metadata["section_title"] = str(chunk.section_title) if chunk.section_title else None
            
            metadatas.append(metadata)
        
        try:
            self.text_collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            if "dimensionality" in str(e).lower():
                logger.error(
                    "Dimension mismatch error. Collection needs to be reset.",
                    error=str(e)
                )
                raise ValueError(
                    f"Embedding dimension mismatch. The collection expects different dimensions. "
                    f"Please run 'python reset_chromadb.py' to reset the collections, "
                    f"or delete the ./chroma_db directory and restart the backend."
                )
            raise
    
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
                metadata["page_number"] = int(chunk.page_number) if chunk.page_number else None
            
            if chunk.section_title:
                metadata["section_title"] = str(chunk.section_title) if chunk.section_title else None
            
            if chunk.table_data:
                # Store essential table metadata
                metadata["table_row_count"] = chunk.table_data.get("row_count", 0)
                metadata["table_column_count"] = chunk.table_data.get("column_count", 0)
                
                # Safely serialize headers (might contain datetime or other non-JSON types)
                try:
                    headers = chunk.table_data.get("headers", [])
                    # Convert any non-string headers to strings
                    headers = [str(h) for h in headers]
                    metadata["table_headers"] = json.dumps(headers)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not serialize table headers: {e}")
                    metadata["table_headers"] = "[]"
                
                # Store full table data separately (ChromaDB has metadata size limits)
                metadata["has_table_data"] = True
            
            metadatas.append(metadata)
        
        try:
            self.table_collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            if "dimensionality" in str(e).lower():
                logger.error(
                    "Dimension mismatch error in table collection. Collection needs to be reset.",
                    error=str(e)
                )
                raise ValueError(
                    f"Embedding dimension mismatch in table collection. "
                    f"Please run 'python reset_chromadb.py' to reset the collections, "
                    f"or delete the ./chroma_db directory and restart the backend."
                )
            raise
    
    async def _add_document_metadata(
        self,
        document_id: str,
        document_name: str,
        chunk_count: int,
        file_size: int = 0
    ):
        """Add document-level metadata"""
        
        # Generate embedding for document name and summary
        summary_text = f"Document: {document_name}, Total chunks: {chunk_count}"
        embeddings = await bedrock_service.generate_embeddings([summary_text])
        
        try:
            self.metadata_collection.add(
                ids=[document_id],
                embeddings=embeddings,
                documents=[summary_text],
                metadatas=[{
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_count": chunk_count,
                    "added_date": datetime.now().isoformat(),
                    "file_size": file_size
                }]
            )
        except Exception as e:
            if "dimensionality" in str(e).lower():
                logger.error(
                    "Dimension mismatch error in metadata collection. Collection needs to be reset.",
                    error=str(e)
                )
                raise ValueError(
                    f"Embedding dimension mismatch in metadata collection. "
                    f"Please run 'python reset_chromadb.py' to reset the collections, "
                    f"or delete the ./chroma_db directory and restart the backend."
                )
            raise
    
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
            logger.info("Generating query embedding", query=query[:100])
            query_embeddings = await bedrock_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            logger.info("Query embedding generated", dimension=len(query_embedding))
            
            results = []
            
            # Semantic search in text collection
            logger.info("Searching text collection", max_results=max_results // 2)
            text_results = await self._semantic_search(
                query_embedding,
                self.text_collection,
                max_results=max_results // 2,
                document_types=document_types,
                retrieval_method="semantic"
            )
            logger.info("Text search completed", results_found=len(text_results))
            results.extend(text_results)
            
            # Semantic search in table collection (if enabled)
            if include_tables:
                logger.info("Searching table collection", max_results=max_results // 2)
                table_results = await self._semantic_search(
                    query_embedding,
                    self.table_collection,
                    max_results=max_results // 2,
                    document_types=document_types,
                    retrieval_method="semantic"
                )
                logger.info("Table search completed", results_found=len(table_results))
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
            logger.error("Semantic search failed", 
                        error=str(e), 
                        collection_name=getattr(collection, 'name', 'unknown'),
                        where_clause=where_clause,
                        max_results=max_results)
            import traceback
            logger.error("Semantic search traceback", traceback=traceback.format_exc())
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
    
    async def list_all_documents(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """List all documents in the system"""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all documents from metadata collection
            all_docs = self.metadata_collection.get()
            
            documents = []
            if all_docs and all_docs['ids']:
                for i, doc_id in enumerate(all_docs['ids']):
                    metadata = all_docs['metadatas'][i]
                    documents.append({
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", "Unknown"),
                        "chunk_count": metadata.get("chunk_count", 0),
                        "added_date": metadata.get("added_date", "Unknown"),
                        "file_size": metadata.get("file_size", 0)  # Will be 0 for now
                    })
            
            # Sort by added_date (newest first)
            documents.sort(key=lambda x: x["added_date"], reverse=True)
            
            # Apply pagination
            total = len(documents)
            documents = documents[skip:skip + limit]
            
            return {
                "documents": documents,
                "total": total,
                "skip": skip,
                "limit": limit
            }
            
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            return {
                "documents": [],
                "total": 0,
                "skip": skip,
                "limit": limit
            }
    
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