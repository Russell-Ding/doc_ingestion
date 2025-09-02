#!/usr/bin/env python3
"""
Simple test to check if segments module can be imported
"""

import sys
import os
import traceback

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

print("🔍 Testing Segments Module Import")
print("="*60)
print(f"Python path includes: {backend_path}\n")

# Step 1: Try importing the segments module directly
print("Step 1: Import segments module")
print("-"*40)
try:
    from app.api.v1.endpoints import segments
    print("✅ Successfully imported segments module")
    print(f"   Module file: {segments.__file__}")
    
    # Check if router exists
    if hasattr(segments, 'router'):
        print("✅ Router found in segments module")
        
        # List all routes
        print("\n   Available routes in segments router:")
        for route in segments.router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                print(f"   - {route.methods} {route.path}")
    else:
        print("❌ No router found in segments module")
        
except Exception as e:
    print(f"❌ Failed to import segments module")
    print(f"   Error: {str(e)}")
    print(f"\n   Full traceback:")
    traceback.print_exc()
    print("\n   This is the root cause of the 405 error!")

# Step 2: Try importing the problematic dependencies
print("\n\nStep 2: Import segments dependencies")
print("-"*40)

dependencies = [
    ("app.core.database", "database"),
    ("app.services.agents", "agents"),
    ("app.services.bedrock", "bedrock"),
    ("app.services.rag_system", "rag_system")
]

for module_name, description in dependencies:
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"✅ {description}: imported successfully")
    except Exception as e:
        print(f"❌ {description}: {str(e)}")
        if "bedrock_utils" in str(e) or "get_runtime" in str(e):
            print(f"   💡 This is a Bedrock configuration issue!")
            print(f"   Solution: Create bedrock_utils.py with get_runtime() function")
            print(f"   Or set USE_DYNAMIC_BEDROCK_RUNTIME=False in config.py")

# Step 3: Try importing the API router
print("\n\nStep 3: Import main API router")
print("-"*40)
try:
    from app.api.v1 import api_router
    print("✅ Successfully imported api_router")
    
    # Check registered routes
    print("\n   Registered routes:")
    for route in api_router.routes:
        if hasattr(route, 'path'):
            path = str(route.path)
            if 'segments' in path:
                methods = route.methods if hasattr(route, 'methods') else 'N/A'
                print(f"   - {methods} {path}")
                
except Exception as e:
    print(f"❌ Failed to import api_router")
    print(f"   Error: {str(e)}")

print("\n" + "="*60)
print("📊 DIAGNOSIS")
print("="*60)

print("""
If the segments module failed to import, the issue is likely:

1. Missing bedrock_utils.py file (if USE_DYNAMIC_BEDROCK_RUNTIME=True)
   → Create the file or set USE_DYNAMIC_BEDROCK_RUNTIME=False

2. Import error in agents.py or its dependencies
   → Check the error messages above

3. Database initialization issues
   → Check database configuration

After fixing the import error:
→ Restart the backend server with: uvicorn app.main:app --reload
""")