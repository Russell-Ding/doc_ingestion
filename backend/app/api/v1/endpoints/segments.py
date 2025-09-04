from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import structlog
import uuid
from datetime import datetime

from app.core.database import get_db
from app.services.agents import report_coordinator_agent

# In-memory storage for segments (since we're not using a real database)
_segments_storage = {}
_segments_by_report = {}

router = APIRouter()
logger = structlog.get_logger(__name__)


class CreateSegmentRequest(BaseModel):
    report_id: str
    name: str
    description: Optional[str] = None
    prompt: str
    order_index: int
    required_document_types: Optional[List[str]] = []
    generation_settings: Optional[Dict[str, Any]] = {}
    generated_content: Optional[str] = None
    content_status: Optional[str] = "pending"
    validation_results: Optional[Dict[str, Any]] = {}


class UpdateSegmentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    order_index: Optional[int] = None
    required_document_types: Optional[List[str]] = None
    generation_settings: Optional[Dict[str, Any]] = None


class SegmentResponse(BaseModel):
    id: str
    report_id: str
    name: str
    description: Optional[str]
    prompt: str
    order_index: int
    content_status: str
    generated_content: Optional[str]
    required_document_types: List[str]
    generation_settings: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]]
    created_date: str
    updated_date: str


class GenerateContentRequest(BaseModel):
    validation_enabled: bool = True


class DirectGenerationRequest(BaseModel):
    segment_data: Dict[str, Any]
    validation_enabled: bool = True
    selected_document_ids: Optional[List[str]] = None


@router.post("/", response_model=SegmentResponse)
async def create_segment(
    request: CreateSegmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new report segment"""
    
    try:
        segment_id = str(uuid.uuid4())
        
        # Store in in-memory storage with actual content
        segment_data = {
            "id": segment_id,
            "report_id": request.report_id,
            "name": request.name,
            "description": request.description,
            "prompt": request.prompt,
            "order_index": request.order_index,
            "content_status": request.content_status or "pending",
            "generated_content": request.generated_content,
            "required_document_types": request.required_document_types or [],
            "generation_settings": request.generation_settings or {},
            "validation_results": request.validation_results or {},
            "created_date": datetime.now().isoformat(),
            "updated_date": datetime.now().isoformat()
        }
        
        # Store in memory
        _segments_storage[segment_id] = segment_data
        
        # Also store by report_id for easy lookup
        if request.report_id not in _segments_by_report:
            _segments_by_report[request.report_id] = []
        _segments_by_report[request.report_id].append(segment_id)
        
        logger.info("Segment created", segment_id=segment_id, name=request.name)
        
        return SegmentResponse(**segment_data)
        
    except Exception as e:
        logger.error("Failed to create segment", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create segment")


@router.get("/report/{report_id}", response_model=List[SegmentResponse])
async def list_segments_for_report(
    report_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all segments for a specific report"""
    
    try:
        # Get segments for this report from in-memory storage
        segment_ids = _segments_by_report.get(report_id, [])
        segments = []
        
        for segment_id in segment_ids:
            if segment_id in _segments_storage:
                segments.append(SegmentResponse(**_segments_storage[segment_id]))
        
        # Sort by order_index
        segments.sort(key=lambda x: x.order_index)
        
        return segments
        
    except Exception as e:
        logger.error("Failed to list segments", error=str(e), report_id=report_id)
        raise HTTPException(status_code=500, detail="Failed to list segments")


@router.get("/{segment_id}", response_model=SegmentResponse)
async def get_segment(
    segment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get segment details"""
    
    try:
        # In a real implementation, query database
        segment_data = {
            "id": segment_id,
            "report_id": "sample-report-id",
            "name": "Sample Segment",
            "description": "Sample description",
            "prompt": "Sample prompt",
            "order_index": 1,
            "content_status": "pending",
            "generated_content": None,
            "required_document_types": [],
            "generation_settings": {},
            "created_date": datetime.now().isoformat(),
            "updated_date": datetime.now().isoformat()
        }
        
        return SegmentResponse(**segment_data)
        
    except Exception as e:
        logger.error("Failed to get segment", error=str(e), segment_id=segment_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve segment")


@router.put("/{segment_id}", response_model=SegmentResponse)
async def update_segment(
    segment_id: str,
    request: UpdateSegmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update segment details"""
    
    try:
        # In a real implementation, update database
        # For now, return updated placeholder data
        
        segment_data = {
            "id": segment_id,
            "report_id": "sample-report-id",
            "name": request.name or "Updated Segment",
            "description": request.description,
            "prompt": request.prompt or "Updated prompt",
            "order_index": request.order_index or 1,
            "content_status": "pending",
            "generated_content": None,
            "required_document_types": request.required_document_types or [],
            "generation_settings": request.generation_settings or {},
            "created_date": datetime.now().isoformat(),
            "updated_date": datetime.now().isoformat()
        }
        
        logger.info("Segment updated", segment_id=segment_id)
        
        return SegmentResponse(**segment_data)
        
    except Exception as e:
        logger.error("Failed to update segment", error=str(e), segment_id=segment_id)
        raise HTTPException(status_code=500, detail="Failed to update segment")


@router.post("/{segment_id}/generate")
async def generate_segment_content(
    segment_id: str,
    request: GenerateContentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate content for a specific segment using AI agents"""
    
    try:
        # In a real implementation, get segment data from database
        segment_data = {
            "name": "Sample Segment",
            "prompt": "Generate a financial summary based on available documents",
            "required_document_types": ["financial_statements"],
            "generation_settings": {}
        }
        
        # Use the report coordinator agent to generate content
        task_data = {
            "segment_data": segment_data,
            "validation_enabled": request.validation_enabled
        }
        
        result = await report_coordinator_agent.execute(task_data)
        
        if not result.success:
            raise HTTPException(
                status_code=500, 
                detail=f"Content generation failed: {result.error_message}"
            )
        
        # In a real implementation, save results to database
        
        logger.info("Segment content generated", segment_id=segment_id)
        
        return {
            "segment_id": segment_id,
            "status": "completed",
            "generated_content": result.data.get("generated_content"),
            "generation_metadata": {
                "execution_time_ms": result.execution_time_ms,
                "documents_used": len(result.data.get("document_retrieval", {}).get("documents_found", [])),
                "validation_results": result.data.get("validation")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate segment content", error=str(e), segment_id=segment_id)
        raise HTTPException(status_code=500, detail="Failed to generate segment content")


@router.post("/generate-report")
async def generate_report_section(
    request: DirectGenerationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate content directly for a report section"""
    
    try:
        # Use the report coordinator agent to generate content
        task_data = {
            "segment_data": request.segment_data,
            "validation_enabled": request.validation_enabled,
            "selected_document_ids": request.selected_document_ids
        }
        
        result = await report_coordinator_agent.execute(task_data)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error_message,
                "generated_content": None
            }
        
        logger.info("Report section generated", segment_name=request.segment_data.get("name"))
        
        return {
            "success": True,
            "generated_content": result.data.get("generated_content"),
            "generation_metadata": {
                "execution_time_ms": result.execution_time_ms,
                "documents_found": len(result.data.get("document_retrieval", {}).get("documents_found", [])),
                "total_chunks": result.data.get("document_retrieval", {}).get("total_chunks", 0),
                "validation_results": result.data.get("validation")
            }
        }
        
    except Exception as e:
        logger.error("Failed to generate report section", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "generated_content": None
        }


@router.get("/{segment_id}/validation")
async def get_segment_validation(
    segment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get validation results for a segment"""
    
    try:
        # In a real implementation, query database for validation results
        return {
            "segment_id": segment_id,
            "validation_results": {
                "overall_quality_score": 0.85,
                "total_issues": 2,
                "issues_by_severity": {
                    "high": 0,
                    "medium": 2,
                    "low": 0,
                    "info": 0
                },
                "validation_comments": [],
                "validation_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error("Failed to get segment validation", error=str(e), segment_id=segment_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve validation results")


@router.delete("/{segment_id}")
async def delete_segment(
    segment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a segment"""
    
    try:
        # In a real implementation, delete from database
        logger.info("Segment deleted", segment_id=segment_id)
        
        return {"message": "Segment deleted successfully"}
        
    except Exception as e:
        logger.error("Failed to delete segment", error=str(e), segment_id=segment_id)
        raise HTTPException(status_code=500, detail="Failed to delete segment")


@router.post("/reorder")
async def reorder_segments(
    segment_orders: List[Dict[str, Any]],
    db: AsyncSession = Depends(get_db)
):
    """Reorder segments within a report"""
    
    try:
        # segment_orders should be: [{"segment_id": "id1", "order_index": 1}, ...]
        
        # In a real implementation, update order_index for each segment in database
        
        logger.info("Segments reordered", count=len(segment_orders))
        
        return {"message": "Segments reordered successfully"}
        
    except Exception as e:
        logger.error("Failed to reorder segments", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reorder segments")