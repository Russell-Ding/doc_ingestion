#!/usr/bin/env python3
"""
Backend API Test Script
Tests all REST API endpoints to ensure the backend is running properly
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

# Backend configuration
API_BASE_URL = "http://localhost:8000/api/v1"
BACKEND_HEALTH_URL = "http://localhost:8000/api/v1/health/"

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def print_result(success: bool, message: str, details: Optional[Dict] = None):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")
    if details:
        print(f"   Details: {json.dumps(details, indent=2)}")
    print()

def make_request(method: str, url: str, **kwargs) -> tuple[bool, Optional[Dict], Optional[str]]:
    """Make HTTP request and return (success, data, error)"""
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
        
        print(f"   {method.upper()} {url}")
        print(f"   Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
        else:
            data = {"raw_response": response.text[:200]}
            print(f"   Response: {response.text[:200]}...")
        
        success = 200 <= response.status_code < 300
        return success, data if success else None, None if success else f"HTTP {response.status_code}: {response.text[:200]}"
        
    except requests.exceptions.ConnectionError:
        return False, None, "Connection refused - Backend server is not running"
    except requests.exceptions.Timeout:
        return False, None, "Request timeout - Backend server is not responding"
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)}"
    except json.JSONDecodeError as e:
        return False, None, f"JSON decode error: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def test_backend_health():
    """Test backend health endpoint"""
    print_test_header("Backend Health Check")
    
    success, data, error = make_request("GET", BACKEND_HEALTH_URL)
    
    if success:
        print_result(True, "Backend server is running and healthy", data)
        return True
    else:
        print_result(False, "Backend server health check failed", {"error": error})
        return False

def test_documents_endpoints():
    """Test document-related endpoints"""
    print_test_header("Document Endpoints")
    
    # Test document listing
    print("📋 Testing document listing...")
    success, data, error = make_request("GET", f"{API_BASE_URL}/documents/")
    
    if success:
        print_result(True, f"Document listing works - found {len(data.get('documents', []))} documents", 
                    {"document_count": len(data.get('documents', []))})
    else:
        print_result(False, "Document listing failed", {"error": error})
        return False
    
    return True

def test_segments_endpoints():
    """Test segment-related endpoints"""
    print_test_header("Segment Endpoints")
    
    # Test the specific endpoint that streamlit uses
    print("🎯 Testing segment report generation (the failing endpoint)...")
    
    test_segment_data = {
        "segment_data": {
            "name": "Executive Summary",
            "prompt": "Provide a high-level executive summary of the financial position and key findings.",
            "required_document_types": [],
            "generation_settings": {}
        },
        "validation_enabled": True
    }
    
    success, data, error = make_request(
        "POST", 
        f"{API_BASE_URL}/segments/generate-report",
        json=test_segment_data,
        headers={"Content-Type": "application/json"}
    )
    
    if success:
        print_result(True, "Segment report generation endpoint works", data)
        
        # Check if content was actually generated
        if data and data.get("generated_content"):
            print_result(True, "Content generation successful", 
                        {"content_preview": data["generated_content"][:100] + "..."})
        else:
            print_result(False, "No content was generated", data)
    else:
        print_result(False, "Segment report generation failed", {"error": error})
        return False
    
    return True

def test_reports_endpoints():
    """Test report-related endpoints"""
    print_test_header("Report Endpoints")
    
    # Test report creation
    print("📝 Testing report creation...")
    
    create_data = {
        "title": "Test Report",
        "description": "Test report for API validation"
    }
    
    success, data, error = make_request(
        "POST",
        f"{API_BASE_URL}/reports/",
        json=create_data,
        headers={"Content-Type": "application/json"}
    )
    
    if success:
        report_id = data.get("id")
        print_result(True, f"Report creation works - created report {report_id}", data)
        
        # Test report listing
        print("📋 Testing report listing...")
        success2, data2, error2 = make_request("GET", f"{API_BASE_URL}/reports/")
        
        if success2:
            print_result(True, f"Report listing works - found {len(data2)} reports", 
                        {"report_count": len(data2)})
        else:
            print_result(False, "Report listing failed", {"error": error2})
            
        return True
    else:
        print_result(False, "Report creation failed", {"error": error})
        return False

def test_bedrock_initialization():
    """Test if Bedrock service is properly initialized by checking any AI-dependent endpoint"""
    print_test_header("Bedrock/AI Service Initialization")
    
    print("🤖 Testing AI service initialization through segment generation...")
    
    # This will test the full pipeline: endpoint -> agent -> bedrock service
    minimal_segment_data = {
        "segment_data": {
            "name": "Test Summary",
            "prompt": "Write a brief test summary.",
            "required_document_types": [],
            "generation_settings": {"max_tokens": 100}
        },
        "validation_enabled": False  # Disable validation to isolate bedrock issues
    }
    
    success, data, error = make_request(
        "POST", 
        f"{API_BASE_URL}/segments/generate-report",
        json=minimal_segment_data,
        headers={"Content-Type": "application/json"}
    )
    
    if success:
        if data.get("success") == False:
            print_result(False, "AI service initialization failed", {"error": data.get("error")})
            return False
        else:
            print_result(True, "AI service (Bedrock) is working properly", 
                        {"content_generated": bool(data.get("generated_content"))})
            return True
    else:
        print_result(False, "AI service test failed at HTTP level", {"error": error})
        return False

def test_full_workflow():
    """Test the complete workflow that Streamlit app uses"""
    print_test_header("Full Streamlit Workflow Test")
    
    print("🔄 Simulating complete streamlit workflow...")
    
    # Step 1: Check documents (streamlit does this)
    print("Step 1: Checking available documents...")
    success1, docs_data, error1 = make_request("GET", f"{API_BASE_URL}/documents/")
    
    if not success1:
        print_result(False, "Workflow failed at document check", {"error": error1})
        return False
    
    print(f"   Found {len(docs_data.get('documents', []))} documents")
    
    # Step 2: Generate Executive Summary (this is what's failing)
    print("Step 2: Generating Executive Summary...")
    executive_summary_data = {
        "segment_data": {
            "name": "Executive Summary",
            "prompt": "Provide a high-level executive summary of the financial position and key findings.",
            "required_document_types": [],
            "generation_settings": {}
        },
        "validation_enabled": True
    }
    
    success2, summary_data, error2 = make_request(
        "POST", 
        f"{API_BASE_URL}/segments/generate-report",
        json=executive_summary_data,
        headers={"Content-Type": "application/json"}
    )
    
    if success2:
        if summary_data.get("success") == False:
            print_result(False, "Executive Summary generation failed", 
                        {"error": summary_data.get("error"), "full_response": summary_data})
            return False
        else:
            print_result(True, "Executive Summary generated successfully", 
                        {"has_content": bool(summary_data.get("generated_content"))})
            return True
    else:
        print_result(False, "Executive Summary generation failed at HTTP level", {"error": error2})
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Backend API Test Suite")
    print(f"Testing backend at: {API_BASE_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Document Endpoints", test_documents_endpoints),
        ("Segment Endpoints", test_segments_endpoints),
        ("Report Endpoints", test_reports_endpoints),
        ("Bedrock/AI Service", test_bedrock_initialization),
        ("Full Streamlit Workflow", test_full_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print_result(False, f"{test_name} test crashed", {"exception": str(e)})
            results[test_name] = False
    
    # Summary
    print_test_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working properly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the details above.")
        
        # Provide specific guidance
        if not results.get("Backend Health"):
            print("\n🔧 NEXT STEPS:")
            print("1. Make sure the backend server is running:")
            print("   cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            print("2. Check if all dependencies are installed:")
            print("   pip install -r requirements.txt")
        elif not results.get("Bedrock/AI Service"):
            print("\n🔧 NEXT STEPS:")
            print("1. Check if bedrock_utils.py exists with get_runtime() function")
            print("2. Verify AWS credentials are properly configured")
            print("3. Check backend logs for Bedrock initialization errors")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)