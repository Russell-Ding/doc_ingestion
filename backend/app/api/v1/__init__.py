from fastapi import APIRouter

from app.api.v1.endpoints import documents, reports, segments, health, enhanced_documents

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(enhanced_documents.router, prefix="/enhanced-documents", tags=["enhanced-documents"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])