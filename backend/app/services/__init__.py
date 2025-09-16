"""
Services initialization module
Ensures services are properly initialized
"""

import structlog

logger = structlog.get_logger(__name__)

# Import services but don't fail if there are issues
try:
    from app.services.bedrock import bedrock_service
    logger.info("Bedrock service imported successfully")
except Exception as e:
    logger.error(f"Failed to import bedrock service: {e}")
    bedrock_service = None

try:
    from app.services.rag_system import rag_system
    logger.info("RAG system imported successfully")
except Exception as e:
    logger.error(f"Failed to import RAG system: {e}")
    rag_system = None

try:
    from app.services.document_processor import DocumentProcessor
    logger.info("Document processor imported successfully")
except Exception as e:
    logger.error(f"Failed to import document processor: {e}")
    DocumentProcessor = None

try:
    from app.services.agents import (
        document_finder_agent,
        content_generator_agent,
        validator_agent,
        report_coordinator_agent
    )
    logger.info("Agents imported successfully")
except Exception as e:
    logger.error(f"Failed to import agents: {e}")
    document_finder_agent = None
    content_generator_agent = None
    validator_agent = None
    report_coordinator_agent = None

async def initialize_services():
    """Initialize all services that require async setup"""
    
    # Initialize Bedrock service
    if bedrock_service:
        try:
            await bedrock_service.initialize()
            logger.info("Bedrock service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock service: {e}")
    
    # Initialize RAG system
    if rag_system:
        try:
            await rag_system.initialize()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    logger.info("All services initialization attempted")