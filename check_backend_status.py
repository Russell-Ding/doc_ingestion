#!/usr/bin/env python3
"""
Quick Backend Status Checker
Simple script to verify if the backend server is running
"""

import requests
import json
import sys

def check_backend_status():
    """Quick check if backend is running"""
    
    print("🔍 Checking Backend Server Status...")
    print("=" * 50)
    
    # Test basic connection
    try:
        response = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
        print(f"✅ Backend server is RUNNING")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Backend server is NOT RUNNING")
        print(f"   Error: Connection refused")
        print(f"\n🔧 To start the backend server:")
        print(f"   cd backend")
        print(f"   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False
        
    except requests.exceptions.Timeout:
        print(f"⚠️  Backend server is SLOW to respond")
        print(f"   Error: Request timeout")
        return False
        
    except Exception as e:
        print(f"❌ Backend server check FAILED")
        print(f"   Error: {str(e)}")
        return False

def check_specific_endpoint():
    """Test the specific endpoint that's failing"""
    
    print(f"\n🎯 Testing Executive Summary Endpoint...")
    print("=" * 50)
    
    test_data = {
        "segment_data": {
            "name": "Executive Summary",
            "prompt": "Provide a brief executive summary.",
            "required_document_types": [],
            "generation_settings": {}
        },
        "validation_enabled": False  # Simplify for testing
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/segments/generate-report",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Endpoint is WORKING")
            
            if data.get("success") == False:
                print(f"❌ But content generation FAILED")
                print(f"   Error: {data.get('error')}")
                return False
            else:
                print(f"✅ Content generation SUCCESSFUL")
                print(f"   Generated content: {len(data.get('generated_content', ''))} characters")
                return True
        else:
            print(f"❌ Endpoint returned ERROR")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Endpoint test FAILED")
        print(f"   Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Backend Status Check")
    print("=" * 50)
    
    # Step 1: Check if server is running
    if not check_backend_status():
        sys.exit(1)
    
    # Step 2: Test specific endpoint
    if not check_specific_endpoint():
        print(f"\n⚠️  Backend is running but Executive Summary generation is failing.")
        print(f"   This suggests an issue with:")
        print(f"   - Bedrock/AWS configuration")
        print(f"   - Missing bedrock_utils.py function")
        print(f"   - Internal backend errors")
        sys.exit(1)
    
    print(f"\n🎉 Everything looks good!")
    print(f"   Backend server: ✅ Running")
    print(f"   Executive Summary: ✅ Working")
    sys.exit(0)