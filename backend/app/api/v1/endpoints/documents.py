from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from pathlib import Path
import aiofiles
import uuid
import os

from app.core.database import get_db
from app.core.config import settings
from app.services.document_processor import document_processor
from app.services.rag_system import rag_system

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a document"""
    
    # Validate file type
    file_extension = Path(file.filename or "").suffix.lower()
    if file_extension not in [f".{ext}" for ext in settings.SUPPORTED_FILE_TYPES]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {settings.SUPPORTED_FILE_TYPES}"
        )
    
    # Validate file size
    content = await file.read()
    file_size = len(content)
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    try:
        # Generate document ID and file path
        document_id = str(uuid.uuid4())
        # Get the base directory (backend folder)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        upload_dir = base_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Store document metadata in database
        # (In a real implementation, you'd use SQLAlchemy models)
        # For now, we'll just process the document
        
        # Process document
        chunks = await document_processor.process_document(
            file_path=str(file_path),
            document_name=document_name or file.filename or "untitled",
            document_id=document_id
        )
        
        # Add to RAG system with file size metadata
        await rag_system.add_document_chunks(
            chunks=chunks,
            document_id=document_id,
            document_name=document_name or file.filename or "untitled",
            file_size=file_size
        )
        
        logger.info(
            "Document uploaded and processed successfully",
            document_id=document_id,
            filename=file.filename,
            chunks_count=len(chunks)
        )
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "chunks_created": len(chunks),
            "processing_status": "completed",
            "message": "Document uploaded and processed successfully"
        }
        
    except Exception as e:
        logger.error("Document upload failed", error=str(e), filename=file.filename)
        
        # Clean up file if it was created
        if 'file_path' in locals() and Path(file_path).exists():
            Path(file_path).unlink()
        
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all uploaded documents"""
    
    try:
        # Get documents from RAG system
        result = await rag_system.list_all_documents(skip=skip, limit=limit)
        
        # Format documents for frontend
        documents = []
        for doc in result["documents"]:
            documents.append({
                "id": doc["document_id"],
                "name": doc["document_name"],
                "chunk_count": doc["chunk_count"],
                "upload_date": doc["added_date"],
                "size": doc.get("file_size", 0),
                "status": "completed"  # All stored docs are completed
            })
        
        return {
            "documents": documents,
            "total": result["total"],
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


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get document details and summary"""
    
    try:
        # Get document summary from RAG system
        summary = await rag_system.get_document_summary(document_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and all its data including files"""
    
    try:
        # Get document info before deletion for file cleanup
        document_summary = await rag_system.get_document_summary(document_id)
        if not document_summary:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from RAG system first
        await rag_system.delete_document(document_id)
        
        # Clean up uploaded file from disk
        try:
            # Look for file with this document_id prefix in uploads directory
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
            upload_dir = base_dir / "uploads"
            
            if upload_dir.exists():
                # Find files that start with the document_id
                for file_path in upload_dir.glob(f"{document_id}_*"):
                    try:
                        file_path.unlink()  # Delete the file
                        logger.info("Deleted file from disk", file_path=str(file_path))
                    except Exception as file_error:
                        logger.warning("Failed to delete file from disk", 
                                     file_path=str(file_path), 
                                     error=str(file_error))
        except Exception as cleanup_error:
            logger.warning("File cleanup failed", error=str(cleanup_error))
            # Don't fail the whole operation if file cleanup fails
        
        logger.info("Document deleted successfully", 
                   document_id=document_id,
                   document_name=document_summary.get("document_name", "Unknown"))
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "document_name": document_summary.get("document_name", "Unknown"),
            "chunks_removed": document_summary.get("total_chunks", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    chunk_type: Optional[str] = None,
    page_number: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get chunks for a specific document"""
    
    try:
        # This would typically query the database for chunks
        # For now, return placeholder
        return {
            "document_id": document_id,
            "chunks": [],
            "filters": {
                "chunk_type": chunk_type,
                "page_number": page_number
            }
        }
        
    except Exception as e:
        logger.error("Failed to get document chunks", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")