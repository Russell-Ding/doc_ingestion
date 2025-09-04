#!/usr/bin/env python3
"""
Script to diagnose backend connectivity issues
"""

import requests
import sys
import os
import subprocess
import socket

def check_port(host, port):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_backend_endpoints():
    """Check various backend endpoints"""
    endpoints = [
        ("http://localhost:8000", "Root"),
        ("http://localhost:8000/health", "Health"),
        ("http://localhost:8000/api/v1/health/", "API Health"),
        ("http://127.0.0.1:8000", "Root (127.0.0.1)"),
        ("http://127.0.0.1:8000/health", "Health (127.0.0.1)"),
    ]
    
    print("🔍 Checking Backend Endpoints:")
    print("-" * 40)
    
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {name}: {url} - OK")
                if "health" in url.lower():
                    print(f"   Response: {response.json()}")
            else:
                print(f"⚠️  {name}: {url} - Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: {url} - Connection refused")
        except requests.exceptions.Timeout:
            print(f"❌ {name}: {url} - Timeout")
        except Exception as e:
            print(f"❌ {name}: {url} - Error: {e}")

def check_processes():
    """Check if uvicorn is running"""
    print("\n🔍 Checking Running Processes:")
    print("-" * 40)
    
    try:
        # Check for uvicorn process
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        uvicorn_processes = [line for line in result.stdout.split('\n') if 'uvicorn' in line and 'grep' not in line]
        
        if uvicorn_processes:
            print("✅ Uvicorn process found:")
            for proc in uvicorn_processes:
                # Extract just the command part
                parts = proc.split()
                if len(parts) > 10:
                    cmd = ' '.join(parts[10:])
                    print(f"   {cmd[:100]}")
        else:
            print("❌ No uvicorn process found")
            
        # Check for python processes on port 8000
        result = subprocess.run(['lsof', '-i', ':8000'], capture_output=True, text=True)
        if result.stdout:
            print("\n✅ Process using port 8000:")
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"   {lines[0]}")  # Header
                print(f"   {lines[1]}")  # First process
        else:
            print("❌ No process found on port 8000")
            
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")

def check_ports():
    """Check if required ports are available"""
    print("\n🔍 Checking Ports:")
    print("-" * 40)
    
    ports = [
        (8000, "Backend (FastAPI)"),
        (8501, "Frontend (Streamlit)"),
    ]
    
    for port, service in ports:
        if check_port("localhost", port):
            print(f"✅ Port {port} ({service}): Open")
        else:
            print(f"❌ Port {port} ({service}): Closed or unavailable")

def suggest_fixes():
    """Suggest fixes based on checks"""
    print("\n💡 Suggested Actions:")
    print("-" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("backend"):
        print("❌ 'backend' directory not found. Are you in the project root?")
        print("   Run: cd /path/to/doc_ingestion")
        return
    
    print("To start the backend, run ONE of these commands:\n")
    
    print("Option 1 (from project root):")
    print("  cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    
    print("Option 2 (from backend directory):")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    
    print("Option 3 (with specific Python):")
    print("  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    
    print("Common issues:")
    print("1. Wrong directory - must run from 'backend' folder")
    print("2. Port already in use - kill existing process:")
    print("   kill -9 $(lsof -t -i:8000)")
    print("3. Missing dependencies - install requirements:")
    print("   pip install -r backend/requirements.txt")
    print("4. Firewall blocking connection")
    print("5. Using wrong URL in frontend (should be http://localhost:8000)")

def main():
    print("🔧 Backend Connectivity Diagnostic Tool")
    print("=" * 40)
    
    check_ports()
    check_backend_endpoints()
    check_processes()
    suggest_fixes()
    
    print("\n" + "=" * 40)
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()