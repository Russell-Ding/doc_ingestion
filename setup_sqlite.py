#!/usr/bin/env python3
"""
SQLite Setup Script for Credit Review System
No PostgreSQL required - uses SQLite + ChromaDB only!
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import aiosqlite
import structlog

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)


def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        logger.info("Command succeeded", command=cmd, output=result.stdout[:200])
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Command failed", command=cmd, error=e.stderr)
        return False


async def create_sqlite_database():
    """Create SQLite database using schema file"""
    try:
        db_path = "credit_reports.db"
        
        # Remove existing database if it exists
        if Path(db_path).exists():
            Path(db_path).unlink()
            logger.info("Removed existing database")
        
        # Read and execute schema
        schema_path = Path("database/sqlite_schema.sql")
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Create database and execute schema
            async with aiosqlite.connect(db_path) as conn:
                await conn.executescript(schema_sql)
                await conn.commit()
            
            logger.info("SQLite database created successfully", path=db_path)
            
            # Verify database creation
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = await cursor.fetchall()
                table_names = [table[0] for table in tables]
                logger.info("Database tables created", tables=table_names)
            
            return True
        else:
            logger.error("Schema file not found", path=str(schema_path))
            return False
        
    except Exception as e:
        logger.error("SQLite database setup failed", error=str(e))
        return False


def setup_python_environment():
    """Setup Python environment and install dependencies"""
    logger.info("Setting up Python environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("Python 3.9+ required")
        return False
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        if not run_command("python -m venv venv"):
            return False
    
    # Install Python dependencies
    pip_cmd = "./venv/bin/pip" if os.name != "nt" else ".\\venv\\Scripts\\pip"
    if not run_command(f"{pip_cmd} install -r backend/requirements.txt"):
        return False
    
    logger.info("Python environment setup complete")
    return True


def setup_node_environment():
    """Setup Node.js environment"""
    logger.info("Setting up Node.js environment...")
    
    # Check if Node.js is installed
    if not run_command("node --version"):
        logger.error("Node.js is required. Please install Node.js 16+")
        return False
    
    # Install frontend dependencies
    if not run_command("npm install", cwd="frontend"):
        return False
    
    logger.info("Node.js environment setup complete")
    return True


def create_env_file():
    """Create environment file for SQLite setup"""
    env_path = Path("backend/.env")
    
    if env_path.exists():
        logger.info("Environment file already exists")
        return True
    
    env_content = """# Database Configuration (SQLite - No PostgreSQL needed!)
DATABASE_URL=sqlite+aiosqlite:///./credit_reports.db
DATABASE_PATH=./credit_reports.db

# AWS Configuration - Dynamic Bedrock Runtime
USE_DYNAMIC_BEDROCK_RUNTIME=true
AWS_REGION=us-east-1

# S3 Bucket for document storage (optional)
S3_BUCKET_NAME=credit-reports-documents

# Application Settings
SECRET_KEY=your-secret-key-here-change-in-production
DEBUG=true
LOG_LEVEL=INFO

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
VECTOR_DB_TYPE=chroma
CHROMA_CLIENT_TYPE=persistent

# Bedrock Models
BEDROCK_TEXT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1

# Redis (optional - for background tasks)
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
BEDROCK_REQUESTS_PER_MINUTE=50
BEDROCK_COST_LIMIT_DAILY=100.0

# File Processing
MAX_FILE_SIZE_MB=100
EXPORT_DIRECTORY=./exports
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info("Environment file created", path=str(env_path))
    logger.warning("IMPORTANT: Create bedrock_utils.py with your get_bedrockruntime function!")
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "backend/chroma_db",
        "backend/exports", 
        "backend/uploads",
        "frontend/public",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directory created", path=directory)
    
    return True


def create_example_bedrock_utils():
    """Create example bedrock_utils.py file"""
    bedrock_utils_path = Path("bedrock_utils.py")
    
    if bedrock_utils_path.exists():
        logger.info("bedrock_utils.py already exists")
        return True
    
    example_content = '''"""
Example bedrock_utils.py - Replace with your implementation
"""

import boto3

def get_bedrockruntime():
    """
    Replace this function with your dynamic credential logic
    
    Example implementations:
    """
    
    # EXAMPLE 1: Static credentials (for testing)
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id='YOUR_ACCESS_KEY',
        aws_secret_access_key='YOUR_SECRET_KEY'
    )
    
    # EXAMPLE 2: Your dynamic credential function would go here
    # dynamic_creds = your_credential_service.get_latest_credentials()
    # return boto3.client(
    #     'bedrock-runtime',
    #     region_name='us-east-1',
    #     aws_access_key_id=dynamic_creds['access_key'],
    #     aws_secret_access_key=dynamic_creds['secret_key']
    # )
'''
    
    with open(bedrock_utils_path, 'w') as f:
        f.write(example_content)
    
    logger.info("Example bedrock_utils.py created", path=str(bedrock_utils_path))
    return True


async def main():
    """Main setup function"""
    logger.info("🚀 Starting Credit Review System Setup (SQLite + ChromaDB)")
    logger.info("✅ No PostgreSQL installation required!")
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up Python environment", setup_python_environment),
        ("Setting up Node.js environment", setup_node_environment), 
        ("Creating environment file", create_env_file),
        ("Creating SQLite database", create_sqlite_database),
        ("Creating example bedrock utils", create_example_bedrock_utils),
    ]
    
    for step_name, step_func in steps:
        logger.info("Running step", step=step_name)
        
        if asyncio.iscoroutinefunction(step_func):
            success = await step_func()
        else:
            success = step_func()
            
        if not success:
            logger.error("Setup failed", step=step_name)
            return False
        
        logger.info("Step completed", step=step_name)
    
    logger.info("🎉 Setup completed successfully!")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. ✅ Database: SQLite database created (credit_reports.db)")
    print("2. ✅ No PostgreSQL needed!")
    print("3. 🔧 Update bedrock_utils.py with your AWS function")
    print("4. 🚀 Start backend: cd backend && uvicorn app.main:app --reload")
    print("5. 🚀 Start frontend: cd frontend && npm start")
    print("6. 🌐 Open http://localhost:3000")
    print("="*60)
    print("📁 Database file: ./credit_reports.db")
    print("📁 ChromaDB data: ./chroma_db/")
    print("📁 Uploads: ./backend/uploads/")
    print("📁 Exports: ./backend/exports/")
    print("="*60)
    

if __name__ == "__main__":
    asyncio.run(main())