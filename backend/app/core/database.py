import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path
import structlog
import aiosqlite

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Create async engine for SQLite
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    # SQLite-specific settings
    connect_args={"check_same_thread": False}
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def init_db():
    """Initialize the SQLite database"""
    try:
        # For SQLite, we'll use the schema file directly
        schema_path = Path("database/sqlite_schema.sql")
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            async with aiosqlite.connect(settings.DATABASE_PATH) as conn:
                await conn.executescript(schema_sql)
                await conn.commit()
            
            logger.info("SQLite database initialized successfully", path=settings.DATABASE_PATH)
        else:
            logger.warning("SQLite schema file not found, using SQLAlchemy metadata", path=str(schema_path))
            # Fallback to SQLAlchemy metadata
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        
    except Exception as e:
        logger.error("Failed to initialize SQLite database", error=str(e))
        raise


async def get_db_session() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Dependency for FastAPI
async def get_db():
    """Database dependency for FastAPI"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()