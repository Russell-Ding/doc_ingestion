#!/usr/bin/env python3
"""
Diagnose 405 Method Not Allowed Error
This script helps identify why the endpoint is returning 405
"""

import requests
import json

def test_endpoint_methods():
    """Test different HTTP methods on the endpoint"""
    
    url = "http://localhost:8000/api/v1/segments/generate-report"
    test_data = {
        "segment_data": {
            "name": "Test",
            "prompt": "Test prompt",
            "required_document_types": [],
            "generation_settings": {}
        },
        "validation_enabled": False
    }
    
    print("🔍 Testing endpoint with different HTTP methods")
    print("=" * 60)
    print(f"Endpoint: {url}\n")
    
    # Test different methods
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    
    for method in methods:
        print(f"Testing {method}...")
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                response = requests.request(
                    method, 
                    url, 
                    json=test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 405:
                # Check Allow header to see what methods are allowed
                allow_header = response.headers.get("Allow", "Not specified")
                print(f"  Allowed methods: {allow_header}")
            elif response.status_code == 200:
                print(f"  ✅ {method} is ACCEPTED")
            else:
                print(f"  Response: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
        print()

def check_openapi_docs():
    """Check OpenAPI documentation for the endpoint"""
    
    print("📚 Checking OpenAPI Documentation")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/api/v1/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi = response.json()
            
            # Look for our endpoint
            path = "/api/v1/segments/generate-report"
            if path in openapi.get("paths", {}):
                methods = openapi["paths"][path]
                print(f"✅ Endpoint found in OpenAPI spec")
                print(f"   Supported methods: {list(methods.keys())}")
                
                # Show POST details if available
                if "post" in methods:
                    print(f"\n   POST endpoint details:")
                    print(f"   - Summary: {methods['post'].get('summary', 'N/A')}")
                    print(f"   - OperationId: {methods['post'].get('operationId', 'N/A')}")
            else:
                print(f"❌ Endpoint NOT found in OpenAPI spec")
                print(f"   This means the endpoint is not registered!")
                
                # List all available segment endpoints
                print(f"\n   Available segment endpoints:")
                for path_key in openapi.get("paths", {}).keys():
                    if "/segments" in path_key:
                        print(f"   - {path_key}")
        else:
            print(f"❌ Could not fetch OpenAPI spec: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching OpenAPI spec: {str(e)}")

def check_fastapi_routes():
    """Try to access FastAPI's automatic documentation"""
    
    print("\n📖 Checking FastAPI Documentation")
    print("=" * 60)
    
    docs_url = "http://localhost:8000/docs"
    print(f"FastAPI Docs URL: {docs_url}")
    print(f"You can open this in your browser to see all available endpoints")
    
    # Check if docs are accessible
    try:
        response = requests.get(docs_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ FastAPI docs are accessible")
        else:
            print(f"❌ FastAPI docs returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not access docs: {str(e)}")

def test_actual_call():
    """Test the actual call that Streamlit makes"""
    
    print("\n🎯 Testing Actual Streamlit Call")
    print("=" * 60)
    
    # Exact data that streamlit sends
    data = {
        "segment_data": {
            "name": "Executive Summary",
            "prompt": "Provide a high-level executive summary of the financial position and key findings.",
            "required_document_types": [],
            "generation_settings": {}
        },
        "validation_enabled": True
    }
    
    url = "http://localhost:8000/api/v1/segments/generate-report"
    
    print(f"POST {url}")
    print(f"Data: {json.dumps(data, indent=2)[:200]}...\n")
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 405:
            print(f"❌ METHOD NOT ALLOWED")
            print(f"   Allow header: {response.headers.get('Allow', 'Not specified')}")
            print(f"   Response: {response.text[:200]}")
        elif response.status_code == 200:
            print(f"✅ Request SUCCESSFUL")
            result = response.json()
            if result.get("success") == False:
                print(f"   But generation failed: {result.get('error')}")
            else:
                print(f"   Content generated: {bool(result.get('generated_content'))}")
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

def main():
    print("🔧 405 Method Not Allowed Error Diagnosis")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/api/v1/health/", timeout=2)
        if response.status_code != 200:
            print("❌ Backend server is not healthy")
            return
        print("✅ Backend server is running\n")
    except:
        print("❌ Backend server is not running!")
        print("   Start it with: cd backend && uvicorn app.main:app --reload")
        return
    
    # Run diagnostic tests
    test_endpoint_methods()
    check_openapi_docs()
    check_fastapi_routes()
    test_actual_call()
    
    print("\n" + "=" * 60)
    print("💡 DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    print("""
If you're getting 405 errors, the most likely causes are:

1. **Backend not reloaded after code changes**
   Solution: Restart the backend server
   
2. **Import error preventing endpoint registration**
   Solution: Check backend logs for import errors
   
3. **URL typo or wrong prefix**
   Solution: Check the OpenAPI docs at http://localhost:8000/docs
   
4. **Method mismatch (GET vs POST)**
   Solution: Ensure using POST method for this endpoint
""")

if __name__ == "__main__":
    main()