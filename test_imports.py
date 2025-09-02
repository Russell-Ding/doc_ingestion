#!/usr/bin/env python3
"""
Test if all backend modules can be imported correctly
This helps identify import errors that prevent endpoints from being registered
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_import(module_path, description):
    """Test importing a module"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Module: {module_path}")
    print('-'*60)
    
    try:
        # Try to import the module
        if module_path.startswith('from '):
            # Handle "from X import Y" style
            exec(module_path)
            print(f"✅ Import successful")
        else:
            # Handle direct imports
            module = __import__(module_path, fromlist=[''])
            print(f"✅ Import successful")
            
            # Check if it's the segments module
            if 'segments' in module_path:
                # Try to access the router
                if hasattr(module, 'router'):
                    print(f"✅ Router found in module")
                    
                    # Check for the specific endpoint
                    router = module.router
                    print(f"   Router routes: {router.routes}")
                    
                    # Look for generate-report endpoint
                    for route in router.routes:
                        if hasattr(route, 'path') and 'generate-report' in str(route.path):
                            print(f"✅ Found /generate-report endpoint!")
                            print(f"   Methods: {route.methods if hasattr(route, 'methods') else 'N/A'}")
                else:
                    print(f"❌ No router found in module")
                    
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        print(f"   This is likely the cause of the 405 error!")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    print("🔍 Backend Import Test")
    print("="*60)
    print("This script tests if all backend modules can be imported correctly.")
    print("Import errors prevent endpoints from being registered, causing 405 errors.")
    
    # Test imports in order of dependency
    tests = [
        ("app.core.config", "Configuration module"),
        ("app.core.database", "Database module"),
        ("app.core.logging", "Logging module"),
        ("app.services.bedrock", "Bedrock service"),
        ("app.services.rag_system", "RAG system"),
        ("app.services.document_processor", "Document processor"),
        ("app.services.agents", "Agent system"),
        ("app.api.v1.endpoints.health", "Health endpoint"),
        ("app.api.v1.endpoints.documents", "Documents endpoint"),
        ("app.api.v1.endpoints.reports", "Reports endpoint"),
        ("app.api.v1.endpoints.segments", "Segments endpoint (THE PROBLEMATIC ONE)"),
        ("from app.api.v1 import api_router", "Main API router"),
        ("app.main", "Main application")
    ]
    
    results = {}
    failed = []
    
    for module_path, description in tests:
        success = test_import(module_path, description)
        results[module_path] = success
        if not success:
            failed.append((module_path, description))
            # Don't continue testing if a critical module fails
            if 'segments' in module_path:
                print("\n⚠️  Stopping tests - segments module failed to import")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 IMPORT TEST SUMMARY")
    print('='*60)
    
    if failed:
        print(f"\n❌ {len(failed)} module(s) failed to import:")
        for module_path, description in failed:
            print(f"   - {description} ({module_path})")
        
        print(f"\n🔧 SOLUTION:")
        print("The import errors above are causing the 405 error.")
        print("Fix these import errors and restart the backend server.")
        
        # Specific guidance for common issues
        if any('bedrock' in f[0] for f in failed):
            print("\n📌 Bedrock import error detected:")
            print("   1. Check if bedrock_utils.py exists with get_runtime() function")
            print("   2. Or set USE_DYNAMIC_BEDROCK_RUNTIME=False in config")
            
        if any('agents' in f[0] for f in failed):
            print("\n📌 Agents import error detected:")
            print("   This is likely due to Bedrock service initialization failing")
            
    else:
        print(f"\n✅ All modules imported successfully!")
        print("   The 405 error might be due to:")
        print("   1. Backend server needs to be restarted")
        print("   2. Wrong URL or method being used")
        print("   3. CORS or middleware issues")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)