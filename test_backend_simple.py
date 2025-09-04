#!/usr/bin/env python3
"""
Simple backend test to see what's actually running
"""

import requests
import json

def test_backend():
    """Test various backend endpoints"""
    
    print("🔍 Testing Backend Connection")
    print("=" * 40)
    
    # Test different possible URLs
    urls_to_test = [
        "http://localhost:8000",
        "http://localhost:8000/",
        "http://localhost:8000/health",
        "http://localhost:8000/api/v1/health/",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/health",
    ]
    
    for url in urls_to_test:
        print(f"\nTesting: {url}")
        try:
            response = requests.get(url, timeout=2)
            print(f"  Status: {response.status_code}")
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"  Response: {json.dumps(data, indent=2)}")
            except:
                print(f"  Response (text): {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection refused - backend not running on this URL")
        except requests.exceptions.Timeout:
            print(f"  ❌ Timeout - backend not responding")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("\n📝 How to start the backend:\n")
    print("1. Open a new terminal")
    print("2. Navigate to project directory:")
    print("   cd /Users/Russell/Library/CloudStorage/Dropbox/MFE_Courses/pytorch/doc_ingestion")
    print("3. Go to backend directory:")
    print("   cd backend")
    print("4. Start the server:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("\n   OR (alternative):")
    print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    test_backend()