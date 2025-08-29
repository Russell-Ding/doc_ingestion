from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from pathlib import Path
import aiofiles
import uuid

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
        upload_dir = Path("backend/uploads")
        upload_dir.mkdir(exist_ok=True)
        
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
        
        # Add to RAG system
        await rag_system.add_document_chunks(
            chunks=chunks,
            document_id=document_id,
            document_name=document_name or file.filename or "untitled"
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
    
    # In a real implementation, this would query the database
    # For now, return a placeholder response
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
    """Delete a document and all its data"""
    
    try:
        # Remove from RAG system
        await rag_system.delete_document(document_id)
        
        # In a real implementation, also delete from database and file storage
        
        logger.info("Document deleted successfully", document_id=document_id)
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error("Failed to delete document", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail="Failed to delete document")


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