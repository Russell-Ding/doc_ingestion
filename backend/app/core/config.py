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
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "credit_app"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "credit_reports"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: Optional[str] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"
    
    # Redis (for caching and background tasks)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # AWS S3 for file storage
    S3_BUCKET_NAME: str = "credit-reports-documents"
    S3_ENDPOINT_URL: Optional[str] = None  # For local MinIO development
    
    # AWS Bedrock Models
    BEDROCK_TEXT_MODEL: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    BEDROCK_EMBEDDING_MODEL: str = "amazon.titan-embed-text-v1"
    BEDROCK_MAX_TOKENS: int = 4000
    BEDROCK_TEMPERATURE: float = 0.7
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FILE_TYPES: List[str] = ["pdf", "docx", "xlsx", "jpg", "jpeg", "png"]
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # RAG Configuration
    VECTOR_DB_TYPE: str = "chroma"  # or "pinecone"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    MAX_RETRIEVED_CHUNKS: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
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