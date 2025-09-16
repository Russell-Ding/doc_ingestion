"""
Test version of segments endpoint to isolate the issue
This is a minimal version without complex dependencies
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import structlog
import json

router = APIRouter()
logger = structlog.get_logger(__name__)


class DirectGenerationRequest(BaseModel):
    segment_data: Dict[str, Any]
    validation_enabled: bool = True


@router.post("/generate-report")
async def generate_report_section(request: DirectGenerationRequest):
    """Test endpoint for report generation"""
    
    logger.info("Test endpoint called", segment_name=request.segment_data.get("name"))
    
    try:
        # First, just return a success message to test if endpoint works
        return {
            "success": True,
            "generated_content": "This is a test response. If you see this, the endpoint is working but the agent system needs to be fixed.",
            "generation_metadata": {
                "test": True,
                "message": "Endpoint is reachable",
                "segment_name": request.segment_data.get("name")
            }
        }
        
    except Exception as e:
        logger.error("Test endpoint failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "generated_content": None
        }


@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Segments router is working", "status": "ok"}