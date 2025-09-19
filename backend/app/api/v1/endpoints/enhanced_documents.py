"""
Enhanced document processing endpoints with LLM summarization for structured data.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import tempfile
import os
from pathlib import Path

from app.core.database import get_db
from app.services.enhanced_document_processor import enhanced_document_processor, EnhancedDocumentChunk
from app.core.config import settings

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/upload-enhanced")
async def upload_enhanced_document(
    file: UploadFile = File(...),
    processing_mode: str = Form("comprehensive"),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process document with enhanced capabilities (CSV, Excel, Email)"""

    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension.lstrip('.') not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {settings.SUPPORTED_FILE_TYPES}"
            )

        # Validate file size
        file_size_mb = len(await file.read()) / (1024 * 1024)
        await file.seek(0)  # Reset file pointer

        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {settings.MAX_FILE_SIZE_MB}MB"
            )

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process document with enhanced processor
            logger.info(
                "Processing enhanced document",
                filename=file.filename,
                size_mb=file_size_mb,
                file_type=file_extension,
                processing_mode=processing_mode
            )

            chunks = await enhanced_document_processor.process_document(
                file_path=temp_file_path,
                document_name=file.filename,
                document_id=f"enhanced_{file.filename}_{int(file_size_mb * 1000)}"
            )

            # Convert chunks to serializable format
            chunk_data = []
            for chunk in chunks:
                chunk_dict = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_type": chunk.chunk_type,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count,
                    "bbox": chunk.bbox,
                    "table_data": chunk.table_data,
                    "summary": getattr(chunk, 'summary', None),
                    "structured_data": getattr(chunk, 'structured_data', None),
                    "email_metadata": getattr(chunk, 'email_metadata', None)
                }
                chunk_data.append(chunk_dict)

            logger.info(
                "Enhanced document processed successfully",
                filename=file.filename,
                chunks_created=len(chunks),
                chunk_types=[chunk.chunk_type for chunk in chunks]
            )

            return {
                "document_id": f"enhanced_{file.filename}",
                "filename": file.filename,
                "file_type": file_extension,
                "file_size_mb": file_size_mb,
                "processing_mode": processing_mode,
                "chunks_created": len(chunks),
                "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
                "chunks": chunk_data,
                "enhanced_features": {
                    "has_table_summaries": any(chunk.chunk_type == "table_summary" for chunk in chunks),
                    "has_email_analysis": any(chunk.chunk_type in ["email_summary", "sender_analysis"] for chunk in chunks),
                    "has_structured_data": any(getattr(chunk, 'structured_data', None) is not None for chunk in chunks),
                    "summary_chunks": len([c for c in chunks if getattr(c, 'summary', None)]),
                    "data_chunks": len([c for c in chunks if c.chunk_type in ["table_data", "csv_data", "email_content"]])
                }
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Enhanced document processing failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/analyze-table")
async def analyze_table_data(
    file: UploadFile = File(...),
    analysis_focus: Optional[str] = Form("comprehensive"),
    db: AsyncSession = Depends(get_db)
):
    """Analyze table data (CSV/Excel) with focused LLM summarization"""

    try:
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in ['.csv', '.xlsx', '.xls']:
            raise HTTPException(
                status_code=400,
                detail="This endpoint only supports table files: CSV, Excel (.xlsx/.xls)"
            )

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process with enhanced processor
            chunks = await enhanced_document_processor.process_document(
                file_path=temp_file_path,
                document_name=file.filename,
                document_id=f"table_analysis_{file.filename}"
            )

            # Extract table-specific analysis
            summary_chunks = [c for c in chunks if c.chunk_type == "table_summary"]
            data_chunks = [c for c in chunks if c.chunk_type in ["table_data", "csv_data"]]
            column_chunks = [c for c in chunks if c.chunk_type == "column_analysis"]

            analysis_result = {
                "filename": file.filename,
                "file_type": file_extension,
                "analysis_focus": analysis_focus,
                "table_summary": summary_chunks[0].summary if summary_chunks else "No summary available",
                "structured_analysis": {
                    "tables_found": len(summary_chunks),
                    "data_chunks": len(data_chunks),
                    "column_analyses": len(column_chunks),
                    "total_chunks": len(chunks)
                },
                "summaries": [
                    {
                        "type": chunk.chunk_type,
                        "title": chunk.section_title,
                        "summary": getattr(chunk, 'summary', chunk.content[:200] + "..."),
                        "structured_data": getattr(chunk, 'structured_data', None)
                    }
                    for chunk in summary_chunks
                ],
                "vector_ready_content": [
                    {
                        "chunk_id": chunk.id,
                        "content_for_search": getattr(chunk, 'summary', chunk.content),
                        "chunk_type": chunk.chunk_type,
                        "metadata": {
                            "section_title": chunk.section_title,
                            "structured_data_available": getattr(chunk, 'structured_data', None) is not None
                        }
                    }
                    for chunk in chunks if getattr(chunk, 'summary', None) or chunk.chunk_type == "table_summary"
                ]
            }

            logger.info(
                "Table analysis completed",
                filename=file.filename,
                tables_analyzed=len(summary_chunks),
                total_chunks=len(chunks)
            )

            return analysis_result

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Table analysis failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Table analysis failed: {str(e)}")


@router.post("/analyze-email")
async def analyze_email_content(
    file: UploadFile = File(...),
    analysis_type: Optional[str] = Form("comprehensive"),
    db: AsyncSession = Depends(get_db)
):
    """Analyze email content with sender analysis and conversation threading"""

    try:
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in ['.eml', '.msg']:
            raise HTTPException(
                status_code=400,
                detail="This endpoint only supports email files: .eml, .msg"
            )

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process with enhanced processor
            chunks = await enhanced_document_processor.process_document(
                file_path=temp_file_path,
                document_name=file.filename,
                document_id=f"email_analysis_{file.filename}"
            )

            # Extract email-specific analysis
            email_summary_chunks = [c for c in chunks if c.chunk_type == "email_summary"]
            sender_analysis_chunks = [c for c in chunks if c.chunk_type == "sender_analysis"]
            conversation_chunks = [c for c in chunks if c.chunk_type in ["email_section", "email_content"]]

            # Extract email metadata
            email_metadata = {}
            if chunks and hasattr(chunks[0], 'email_metadata') and chunks[0].email_metadata:
                email_metadata = chunks[0].email_metadata

            analysis_result = {
                "filename": file.filename,
                "file_type": file_extension,
                "analysis_type": analysis_type,
                "email_metadata": email_metadata,
                "email_summary": email_summary_chunks[0].summary if email_summary_chunks else "No summary available",
                "sender_analysis": sender_analysis_chunks[0].content if sender_analysis_chunks else "No sender analysis available",
                "conversation_structure": {
                    "total_sections": len(conversation_chunks),
                    "email_summary_chunks": len(email_summary_chunks),
                    "sender_analysis_chunks": len(sender_analysis_chunks),
                    "conversation_chunks": len(conversation_chunks)
                },
                "conversation_sections": [
                    {
                        "section_index": getattr(chunk.email_metadata, 'section_index', i) if hasattr(chunk, 'email_metadata') and chunk.email_metadata else i,
                        "title": chunk.section_title,
                        "content_preview": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
                        "chunk_type": chunk.chunk_type
                    }
                    for i, chunk in enumerate(conversation_chunks)
                ],
                "vector_ready_content": [
                    {
                        "chunk_id": chunk.id,
                        "content_for_search": getattr(chunk, 'summary', chunk.content),
                        "chunk_type": chunk.chunk_type,
                        "metadata": {
                            "section_title": chunk.section_title,
                            "email_metadata": getattr(chunk, 'email_metadata', None)
                        }
                    }
                    for chunk in chunks
                ]
            }

            logger.info(
                "Email analysis completed",
                filename=file.filename,
                sections_found=len(conversation_chunks),
                total_chunks=len(chunks)
            )

            return analysis_result

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Email analysis failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Email analysis failed: {str(e)}")


@router.get("/supported-formats")
async def get_supported_enhanced_formats():
    """Get information about supported file formats and their enhanced capabilities"""

    return {
        "supported_formats": {
            "csv": {
                "description": "Comma-separated values with LLM table summarization",
                "features": ["table_summary", "column_analysis", "data_patterns", "vector_search_ready"],
                "chunk_types": ["table_summary", "table_data", "column_analysis"],
                "use_cases": ["financial_data", "sales_reports", "customer_data", "metrics"]
            },
            "xlsx": {
                "description": "Excel files with sheet-by-sheet LLM analysis",
                "features": ["multi_sheet_support", "table_summary", "structured_data", "vector_search_ready"],
                "chunk_types": ["table_summary", "table_data"],
                "use_cases": ["financial_statements", "budget_analysis", "data_reports"]
            },
            "xls": {
                "description": "Legacy Excel files with enhanced processing",
                "features": ["table_summary", "structured_data", "vector_search_ready"],
                "chunk_types": ["table_summary", "table_data"],
                "use_cases": ["legacy_reports", "historical_data"]
            },
            "eml": {
                "description": "Standard email files with sender analysis",
                "features": ["sender_analysis", "conversation_threading", "email_metadata", "vector_search_ready"],
                "chunk_types": ["email_summary", "sender_analysis", "email_section"],
                "use_cases": ["communication_analysis", "decision_tracking", "correspondence_review"]
            },
            "msg": {
                "description": "Outlook email files with enhanced analysis",
                "features": ["outlook_metadata", "sender_analysis", "conversation_threading", "vector_search_ready"],
                "chunk_types": ["email_summary", "sender_analysis", "email_section"],
                "use_cases": ["corporate_communications", "email_discovery", "correspondence_analysis"]
            }
        },
        "enhanced_capabilities": {
            "llm_summarization": "Tables and emails are summarized by AI for better vector search",
            "structured_data_preservation": "Original data is preserved alongside summaries",
            "vector_search_optimization": "Content is optimized for semantic search and retrieval",
            "intelligent_chunking": "Content is split intelligently based on data type",
            "metadata_extraction": "Rich metadata is extracted for filtering and organization"
        },
        "api_endpoints": {
            "/upload-enhanced": "Process any supported file with enhanced capabilities",
            "/analyze-table": "Focused analysis for CSV/Excel files",
            "/analyze-email": "Focused analysis for email files",
            "/supported-formats": "This endpoint - format information"
        }
    }