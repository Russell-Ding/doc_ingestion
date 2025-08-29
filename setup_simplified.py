#!/usr/bin/env python3
"""
Simplified Setup Script for Credit Review System
No pgvector dependency - uses ChromaDB only
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import asyncpg
import structlog

# Setup logging
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


async def create_database():
    """Create the database and run schema"""
    try:
        # Connect to default database to create our database
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        
        # Create database if it doesn't exist
        try:
            await conn.execute("CREATE DATABASE credit_reports")
            logger.info("Database 'credit_reports' created")
        except asyncpg.exceptions.DuplicateDatabaseError:
            logger.info("Database 'credit_reports' already exists")
        
        await conn.close()
        
        # Connect to our database and run schema
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database="credit_reports"
        )
        
        # Read and execute schema
        schema_path = Path("database/simple_schema.sql")
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            await conn.execute(schema_sql)
            logger.info("Database schema created successfully")
        else:
            logger.error("Schema file not found", path=str(schema_path))
            return False
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
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
    """Create environment file with default values"""
    env_path = Path("backend/.env")
    
    if env_path.exists():
        logger.info("Environment file already exists")
        return True
    
    env_content = """# Database Configuration
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=credit_reports
POSTGRES_PORT=5432

# AWS Configuration - Two modes available:
# Mode 1: Dynamic Bedrock Runtime (RECOMMENDED for dynamic keys)
USE_DYNAMIC_BEDROCK_RUNTIME=true
# BEDROCK_RUNTIME_FUNCTION_PATH=  # Optional: path to your function file

# Mode 2: Static AWS Credentials (set USE_DYNAMIC_BEDROCK_RUNTIME=false to use)
AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your_aws_access_key_here
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

# S3 Bucket for document storage
S3_BUCKET_NAME=credit-reports-documents

# Application Settings
SECRET_KEY=your-secret-key-here-change-in-production
DEBUG=true
LOG_LEVEL=INFO

# ChromaDB Configuration (no pgvector needed!)
CHROMA_PERSIST_DIRECTORY=./chroma_db
VECTOR_DB_TYPE=chroma

# Bedrock Models
BEDROCK_TEXT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1

# Redis (optional - for background tasks)
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
BEDROCK_REQUESTS_PER_MINUTE=50
BEDROCK_COST_LIMIT_DAILY=100.0
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


async def main():
    """Main setup function"""
    logger.info("Starting Credit Review System Setup (ChromaDB-only version)")
    
    # Check prerequisites
    logger.info("Checking prerequisites...")
    
    # Check if PostgreSQL is running
    if not run_command("pg_isready"):
        logger.error("PostgreSQL is not running. Please start PostgreSQL service.")
        return False
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up Python environment", setup_python_environment),
        ("Setting up Node.js environment", setup_node_environment), 
        ("Creating environment file", create_env_file),
        ("Creating database and schema", create_database),
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
    print("1. Update AWS credentials in backend/.env")
    print("2. Start the backend: cd backend && uvicorn app.main:app --reload")
    print("3. Start the frontend: cd frontend && npm start")
    print("4. Open http://localhost:3000 in your browser")
    print("="*60)
    print("\nNOTE: This version uses ChromaDB only - no pgvector extension required!")
    

if __name__ == "__main__":
    asyncio.run(main())