#!/usr/bin/env python3
"""
Script to fix the segments endpoint 405 error
This helps switch between test and real implementation
"""

import os
import sys

def switch_to_test_endpoint():
    """Switch to test endpoint to verify routing works"""
    
    init_file = "backend/app/api/v1/__init__.py"
    
    test_content = """from fastapi import APIRouter

from app.api.v1.endpoints import documents, reports, segments_test, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(segments_test.router, prefix="/segments", tags=["segments"])  # Using test version
"""
    
    with open(init_file, 'w') as f:
        f.write(test_content)
    
    print("✅ Switched to TEST segments endpoint")
    print("   This endpoint will always return a test response")
    print("   If this works, the issue is in the agents/bedrock initialization")

def switch_to_real_endpoint():
    """Switch back to real endpoint"""
    
    init_file = "backend/app/api/v1/__init__.py"
    
    real_content = """from fastapi import APIRouter

from app.api.v1.endpoints import documents, reports, segments, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])
"""
    
    with open(init_file, 'w') as f:
        f.write(real_content)
    
    print("✅ Switched to REAL segments endpoint")
    print("   This uses the full agent system with Bedrock")

def check_current_setup():
    """Check which endpoint is currently active"""
    
    init_file = "backend/app/api/v1/__init__.py"
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    if 'segments_test' in content:
        print("📍 Currently using: TEST segments endpoint")
    else:
        print("📍 Currently using: REAL segments endpoint")

def main():
    print("🔧 Segments Endpoint Fix Tool")
    print("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            switch_to_test_endpoint()
        elif command == "real":
            switch_to_real_endpoint()
        elif command == "check":
            check_current_setup()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python fix_segments_endpoint.py [test|real|check]")
    else:
        # Interactive mode
        check_current_setup()
        
        print("\nOptions:")
        print("1. Switch to TEST endpoint (to verify routing works)")
        print("2. Switch to REAL endpoint (full functionality)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            switch_to_test_endpoint()
        elif choice == "2":
            switch_to_real_endpoint()
        elif choice == "3":
            print("Exiting...")
        else:
            print("Invalid choice")
    
    print("\n⚠️  IMPORTANT: After switching, restart the backend server:")
    print("   cd backend && uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()