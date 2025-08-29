from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.database import get_db
from app.services.bedrock import bedrock_service
from app.services.rag_system import rag_system

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Credit Review Report Generation API",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check with service dependencies"""
    
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {}
    }
    
    # Check database connection
    try:
        await db.execute("SELECT 1")
        health_status["services"]["database"] = {"status": "healthy"}
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        health_status["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Bedrock service
    try:
        usage_stats = bedrock_service.get_usage_stats()
        health_status["services"]["bedrock"] = {
            "status": "healthy",
            "usage": usage_stats
        }
    except Exception as e:
        logger.error("Bedrock health check failed", error=str(e))
        health_status["services"]["bedrock"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check RAG system
    try:
        if rag_system._initialized:
            health_status["services"]["rag_system"] = {"status": "healthy"}
        else:
            health_status["services"]["rag_system"] = {"status": "initializing"}
    except Exception as e:
        logger.error("RAG system health check failed", error=str(e))
        health_status["services"]["rag_system"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    return health_status