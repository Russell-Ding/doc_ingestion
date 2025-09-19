from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import secrets


class Settings(BaseSettings):
    """Application settings"""
    
    # Project Info
    PROJECT_NAME: str = "Credit Review Report Generation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database (SQLite - no PostgreSQL needed!)
    DATABASE_URL: str = "sqlite+aiosqlite:///./credit_reports.db"
    DATABASE_PATH: str = "./credit_reports.db"
    
    # Redis (for caching and background tasks)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # Dynamic Bedrock Runtime Configuration
    USE_DYNAMIC_BEDROCK_RUNTIME: bool = True  # Set to False to use static AWS credentials
    BEDROCK_RUNTIME_FUNCTION_PATH: Optional[str] = None  # Path to your function module
    
    # AWS S3 for file storage
    S3_BUCKET_NAME: str = "credit-reports-documents"
    S3_ENDPOINT_URL: Optional[str] = None  # For local MinIO development
    
    # AWS Bedrock Models
    BEDROCK_TEXT_MODEL: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    BEDROCK_EMBEDDING_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_EMBEDDING_DIMENSION: int = 1024  # Titan Embed v2 dimension (same as v1)
    BEDROCK_MAX_TOKENS: int = 4000
    BEDROCK_TEMPERATURE: float = 0.7
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FILE_TYPES: List[str] = ["pdf", "docx", "xlsx", "xls", "csv", "eml", "msg", "jpg", "jpeg", "png"]
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Large File Processing Limits
    MAX_EXCEL_CONTENT_LENGTH: int = 10000  # Max chars for Excel content before truncation
    MAX_EXCEL_ROWS_FOR_FULL_DATA: int = 50  # Max rows to include full data
    MAX_EMBEDDING_TEXT_LENGTH: int = 49000  # Max chars for embedding (Bedrock limit is 50k)
    
    # RAG Configuration  
    VECTOR_DB_TYPE: str = "chroma"  # ChromaDB only (no pgvector dependency)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    MAX_RETRIEVED_CHUNKS: int = 5  # Final number after re-ranking
    INITIAL_RETRIEVAL_COUNT: int = 50  # Initial candidates before re-ranking
    SIMILARITY_THRESHOLD: float = 0.3  # Lower threshold for initial retrieval
    RERANK_THRESHOLD: float = 0.7  # Higher threshold for final results after re-ranking
    
    # ChromaDB Settings
    CHROMA_HOST: str = "localhost" 
    CHROMA_PORT: int = 8000
    CHROMA_CLIENT_TYPE: str = "persistent"  # "persistent" or "http"
    
    # Content Generation
    ENABLE_CONTENT_VALIDATION: bool = True
    VALIDATION_CONFIDENCE_THRESHOLD: float = 0.8
    MAX_GENERATION_RETRIES: int = 3
    
    # Export Settings
    EXPORT_DIRECTORY: str = "./exports"
    WORD_TEMPLATE_PATH: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    STRUCTURED_LOGGING: bool = True
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Rate Limiting (for AWS Bedrock)
    BEDROCK_REQUESTS_PER_MINUTE: int = 50
    BEDROCK_COST_LIMIT_DAILY: float = 100.0  # USD
    
    # Development
    DEBUG: bool = False
    TESTING: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()