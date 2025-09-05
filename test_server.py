#!/usr/bin/env python3
"""
Test script to verify Document Portal API endpoints and functionality.
Run this after starting the server with: uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8080"

def test_health_endpoints():
    """Test health check endpoints."""
    print("🔍 Testing Health Endpoints...")
    
    # Basic health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Basic Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Basic Health Failed: {e}")
    
    # Detailed health check
    try:
        response = requests.get(f"{BASE_URL}/health/detailed")
        print(f"✅ Detailed Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Detailed Health Failed: {e}")

def test_analytics_endpoints():
    """Test analytics endpoints."""
    print("\n📊 Testing Analytics Endpoints...")
    
    # Cache stats
    try:
        response = requests.get(f"{BASE_URL}/analytics/cache-stats")
        print(f"✅ Cache Stats: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Cache Data: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"❌ Cache Stats Failed: {e}")
    
    # Token usage
    try:
        response = requests.get(f"{BASE_URL}/analytics/token-usage")
        print(f"✅ Token Usage: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Token Data: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"❌ Token Usage Failed: {e}")

def test_ui_endpoint():
    """Test UI serving endpoint."""
    print("\n🌐 Testing UI Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ UI Endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Content Length: {len(response.text)} characters")
            if "Analytics" in response.text:
                print("   ✅ Analytics tab found in UI")
            if "Document Analysis" in response.text:
                print("   ✅ Document Analysis tab found in UI")
    except Exception as e:
        print(f"❌ UI Endpoint Failed: {e}")

def test_document_analysis():
    """Test document analysis with a sample text file."""
    print("\n📄 Testing Document Analysis...")
    
    # Create a sample text file for testing
    sample_content = """
    This is a sample document for testing the enhanced document portal.
    
    Key Features:
    - Multi-format document support
    - Caching system for improved performance
    - Token usage tracking
    - Evaluation metrics
    - Memory management
    
    The document portal now supports various file formats including PDF, DOCX, TXT, MD, and more.
    """
    
    sample_file = Path("test_sample.txt")
    sample_file.write_text(sample_content)
    
    try:
        with open(sample_file, 'rb') as f:
            files = {'file': ('test_sample.txt', f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/analyze", files=files)
            print(f"✅ Document Analysis: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Analysis Keys: {list(data.keys())}")
    except Exception as e:
        print(f"❌ Document Analysis Failed: {e}")
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()

def run_comprehensive_test():
    """Run all tests."""
    print("🚀 Starting Document Portal API Tests")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    test_health_endpoints()
    test_analytics_endpoints()
    test_ui_endpoint()
    test_document_analysis()
    
    print("\n" + "=" * 50)
    print("✅ Testing Complete!")
    print("\n📋 Next Steps:")
    print("1. Open browser and go to: http://localhost:8080")
    print("2. Test the Analytics tab for cache stats and token usage")
    print("3. Upload documents in the Document Analysis tab")
    print("4. Try the chat functionality with document indexing")
    print("5. Check evaluation metrics and quality scores")

if __name__ == "__main__":
    run_comprehensive_test()
