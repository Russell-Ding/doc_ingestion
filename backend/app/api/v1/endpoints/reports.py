from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import structlog
import uuid
from datetime import datetime
from pathlib import Path

from app.core.database import get_db
from app.services.agents import report_coordinator_agent
from app.services.word_export import word_export_service

# In-memory storage for reports and segments (since we're not using a real database)
_reports_storage = {}
_segments_storage = {}

router = APIRouter()
logger = structlog.get_logger(__name__)


class CreateReportRequest(BaseModel):
    title: str
    description: Optional[str] = None


class GenerateReportRequest(BaseModel):
    segments: List[str]  # List of segment IDs
    validation_enabled: bool = True
    export_format: Optional[str] = "word"


class ReportResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    status: str
    created_date: str
    segments_count: int


@router.post("/", response_model=ReportResponse)
async def create_report(
    request: CreateReportRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new report"""
    
    try:
        report_id = str(uuid.uuid4())
        
        # Store in in-memory storage
        report_data = {
            "id": report_id,
            "title": request.title,
            "description": request.description,
            "status": "draft",
            "created_date": datetime.now().isoformat(),
            "segments_count": 0
        }
        
        _reports_storage[report_id] = report_data
        
        logger.info("Report created", report_id=report_id, title=request.title)
        
        return ReportResponse(**report_data)
        
    except Exception as e:
        logger.error("Failed to create report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create report")


@router.get("/", response_model=List[ReportResponse])
async def list_reports(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all reports"""
    
    # In a real implementation, query database with filters
    # For now, return empty list
    return []


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get report details"""
    
    try:
        # In a real implementation, query database
        # For now, return placeholder
        report_data = {
            "id": report_id,
            "title": "Sample Report",
            "description": "Sample description",
            "status": "draft",
            "created_date": datetime.now().isoformat(),
            "segments_count": 0
        }
        
        return ReportResponse(**report_data)
        
    except Exception as e:
        logger.error("Failed to get report", error=str(e), report_id=report_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve report")


@router.post("/{report_id}/generate")
async def generate_report(
    report_id: str,
    request: GenerateReportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Generate report content using AI agents"""
    
    try:
        # Start background task for report generation
        background_tasks.add_task(
            _generate_report_background,
            report_id,
            request.segments,
            request.validation_enabled
        )
        
        logger.info(
            "Report generation started",
            report_id=report_id,
            segments_count=len(request.segments)
        )
        
        return {
            "message": "Report generation started",
            "report_id": report_id,
            "status": "generating"
        }
        
    except Exception as e:
        logger.error("Failed to start report generation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start report generation")


@router.get("/{report_id}/status")
async def get_report_status(
    report_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get report generation status"""
    
    try:
        # In a real implementation, query database for current status
        return {
            "report_id": report_id,
            "status": "draft",
            "progress": 0,
            "current_segment": None,
            "total_segments": 0,
            "message": "Report ready for generation"
        }
        
    except Exception as e:
        logger.error("Failed to get report status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get report status")


@router.post("/{report_id}/export/{format}")
async def export_report(
    report_id: str,
    format: str,
    include_validations: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Export report to specified format"""
    
    if format not in ["word", "pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    try:
        # Get report data from in-memory storage
        if report_id not in _reports_storage:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_data = _reports_storage[report_id]
        
        # Import the segments storage from segments.py
        from app.api.v1.endpoints.segments import _segments_by_report, _segments_storage
        
        # Get segments for this report from in-memory storage
        segment_ids = _segments_by_report.get(report_id, [])
        segments = []
        validation_results = []
        
        for segment_id in segment_ids:
            if segment_id in _segments_storage:
                segment_data = _segments_storage[segment_id]
                segments.append(segment_data)
                
                # Get validation results - ensure it has the validation_issues for Word comments
                validation_data = segment_data.get('validation_results', {})
                
                # If validation data exists but doesn't have word_comments, generate them
                if validation_data and not validation_data.get('word_comments'):
                    # Check for validation_issues or issues field
                    validation_issues = validation_data.get('validation_issues') or validation_data.get('issues', [])
                    if validation_issues:
                        word_comments = []
                        content = segment_data.get('generated_content', '')
                        
                        for i, issue in enumerate(validation_issues):
                            # Get the text span that this issue refers to
                            text_span = issue.get('text_span', issue.get('text', ''))
                            
                            # If no text_span, try to extract from content based on issue description
                            if not text_span and content:
                                # Use first sentence of content as fallback
                                sentences = content.split('. ')
                                text_span = sentences[0] if sentences else content[:100]
                            
                            comment = {
                                "id": f"validation_{segment_id}_{i+1}",
                                "text": f"[{issue.get('issue_type', 'Issue').upper()}] {issue.get('description', 'No description')}",
                                "start": 0,  # Will be calculated by Word export service
                                "end": len(text_span) if text_span else 1,
                                "author": "AI Validator",
                                "date": datetime.now().isoformat(),
                                "severity": issue.get('severity', 'medium'),
                                "type": issue.get('issue_type', 'validation'),
                                "text_span": text_span  # Add this for the Word service to find
                            }
                            
                            if issue.get('suggested_fix'):
                                comment["text"] += f"\n\nSuggested Fix: {issue.get('suggested_fix')}"
                            
                            word_comments.append(comment)
                        
                        validation_data['word_comments'] = word_comments
                        
                        logger.info(f"Generated {len(word_comments)} word comments for segment {segment_id}")
                    else:
                        logger.info(f"No validation issues found for segment {segment_id}")
                
                validation_results.append(validation_data)
        
        # Sort segments by order_index
        segments.sort(key=lambda x: x.get('order_index', 0))
        
        logger.info(f"Exporting report with {len(segments)} segments", 
                   report_id=report_id, 
                   report_title=report_data.get('title'))
        
        if format == "word":
            export_result = await word_export_service.generate_report(
                report_data=report_data,
                segments=segments,
                validation_results=validation_results,
                include_validation_comments=include_validations
            )
            
            return {
                "export_type": format,
                "filename": export_result["filename"],
                "download_url": export_result["download_url"],
                "file_size": export_result["file_size"]
            }
        
        # PDF export would be implemented here
        raise HTTPException(status_code=501, detail="PDF export not yet implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export report")


@router.get("/download/{filename}")
async def download_report(filename: str):
    """Download a generated report file"""
    
    try:
        from app.core.config import settings
        
        # Construct the file path
        export_directory = Path(settings.EXPORT_DIRECTORY)
        file_path = export_directory / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.error("Report file not found", filename=filename, path=str(file_path))
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Security check: ensure the file is within the export directory
        if not str(file_path.resolve()).startswith(str(export_directory.resolve())):
            logger.error("Invalid file path", filename=filename)
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info("Downloading report file", filename=filename, file_size=file_path.stat().st_size)
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download report", error=str(e), filename=filename)
        raise HTTPException(status_code=500, detail="Failed to download report")


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a report and all its data"""
    
    try:
        # In a real implementation, delete from database
        logger.info("Report deleted", report_id=report_id)
        
        return {"message": "Report deleted successfully"}
        
    except Exception as e:
        logger.error("Failed to delete report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete report")


async def _generate_report_background(
    report_id: str,
    segment_ids: List[str],
    validation_enabled: bool
):
    """Background task for report generation"""
    
    try:
        logger.info("Starting background report generation", report_id=report_id)
        
        # In a real implementation, this would:
        # 1. Get segment data from database
        # 2. Use report_coordinator_agent to generate content
        # 3. Update database with results
        # 4. Handle validation if enabled
        
        # Placeholder for actual implementation
        for i, segment_id in enumerate(segment_ids):
            logger.info(
                "Processing segment",
                report_id=report_id,
                segment_id=segment_id,
                progress=f"{i+1}/{len(segment_ids)}"
            )
            
            # Would call: await report_coordinator_agent.execute(segment_data)
        
        logger.info("Background report generation completed", report_id=report_id)
        
    except Exception as e:
        logger.error("Background report generation failed", error=str(e), report_id=report_id)