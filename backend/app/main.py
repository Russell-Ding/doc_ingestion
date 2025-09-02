from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.core.config import settings
from app.core.database import init_db
from app.api.v1 import api_router
from app.core.logging import setup_logging

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Credit Review Report Generation System")
    
    # Initialize database
    await init_db()
    
    # Initialize services
    try:
        from app.services import initialize_services
        await initialize_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue anyway - some endpoints might still work
    
    yield
    
    logger.info("Shutting down Credit Review Report Generation System")


def create_application() -> FastAPI:
    """Create FastAPI application with all configurations"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Credit Review Report Generation System with AI-powered document analysis",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan
    )
    
    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    return app


app = create_application()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Credit Review Report Generation System API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME
    }