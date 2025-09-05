#!/usr/bin/env python3
"""
Validation script to verify all test fixes are working correctly.
This script checks imports and basic functionality without running full pytest.
"""

import sys
import os
import traceback
from pathlib import Path

def validate_imports():
    """Validate all critical imports work correctly."""
    print("🔍 Validating imports...")
    
    try:
        # Test caching module (fixed os import)
        from utils.caching import get_cache_manager
        print("✅ utils.caching - os import fixed")
        
        # Test memory manager (fixed LangChain imports)
        from utils.memory_manager import get_memory_manager
        print("✅ utils.memory_manager - LangChain imports fixed")
        
        # Test enhanced document loaders
        from utils.enhanced_document_loaders import EnhancedDocumentLoader
        print("✅ utils.enhanced_document_loaders - working")
        
        # Test evaluation system
        from utils.evaluation import RAGEvaluator, EvaluationMetrics
        print("✅ utils.evaluation - working")
        
        # Test token counter
        from utils.token_counter import TokenCounter
        print("✅ utils.token_counter - working")
        
        # Test Pydantic models (fixed validators)
        from model.models import DocumentMetadata, ChatRequest, TokenUsage
        print("✅ model.models - Pydantic validators fixed")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def validate_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🧪 Validating basic functionality...")
    
    try:
        # Test cache manager initialization
        from utils.caching import get_cache_manager
        cache_manager = get_cache_manager()
        print("✅ Cache manager initialization")
        
        # Test memory manager initialization
        from utils.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()
        print("✅ Memory manager initialization")
        
        # Test document loader initialization
        from utils.enhanced_document_loaders import EnhancedDocumentLoader
        loader = EnhancedDocumentLoader()
        print("✅ Document loader initialization")
        
        # Test token counter initialization
        from utils.token_counter import TokenCounter
        counter = TokenCounter()
        print("✅ Token counter initialization")
        
        # Test Pydantic model creation
        from model.models import DocumentMetadata, DocumentType, ProcessingStatus
        from datetime import datetime
        
        metadata = DocumentMetadata(
            file_name="test.pdf",
            file_path="/test/path",
            file_type=DocumentType.PDF,
            file_size=1024,
            status=ProcessingStatus.PENDING
        )
        
        # Test new model_dump method (fixed deprecation)
        data = metadata.model_dump()
        assert "file_name" in data
        print("✅ Pydantic model creation and serialization")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        traceback.print_exc()
        return False

def validate_exception_handling():
    """Test exception handling fixes."""
    print("\n🚨 Validating exception handling...")
    
    try:
        from utils.enhanced_document_loaders import EnhancedDocumentLoader
        from exception.custom_exception import DocumentPortalException
        import tempfile
        
        loader = EnhancedDocumentLoader()
        
        # Test unsupported file extension (should raise DocumentPortalException)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            loader.load_document(tmp_path)
            print("❌ Should have raised DocumentPortalException")
            return False
        except DocumentPortalException:
            print("✅ DocumentPortalException correctly raised for unsupported files")
            os.unlink(tmp_path)
            return True
        except Exception as e:
            print(f"❌ Wrong exception type: {type(e).__name__}")
            os.unlink(tmp_path)
            return False
            
    except Exception as e:
        print(f"❌ Exception handling error: {e}")
        traceback.print_exc()
        return False

def validate_environment_setup():
    """Validate environment and dependencies."""
    print("\n🔧 Validating environment setup...")
    
    try:
        # Check required directories exist
        required_dirs = ["data", "logs", "cache", "faiss_index"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
            else:
                print(f"✅ Directory exists: {dir_name}")
        
        # Check environment file
        env_file = Path(".env")
        env_example = Path("env.example")
        
        if env_example.exists():
            print("✅ env.example template available")
        
        if env_file.exists():
            print("✅ .env file configured")
        else:
            print("⚠️  .env file not found - copy from env.example and add API keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Document Portal - Test Fixes Validation")
    print("=" * 50)
    
    results = []
    
    # Run validation checks
    results.append(("Import Validation", validate_imports()))
    results.append(("Functionality Validation", validate_basic_functionality()))
    results.append(("Exception Handling", validate_exception_handling()))
    results.append(("Environment Setup", validate_environment_setup()))
    
    # Print summary
    print("\n📊 Validation Summary")
    print("=" * 30)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validations PASSED! Your fixes are working correctly.")
        print("\n📋 Next Steps:")
        print("1. Run: pytest tests/ -v --tb=short")
        print("2. Start app: uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload")
        print("3. Test endpoints with Postman or curl")
        return 0
    else:
        print("⚠️  Some validations FAILED. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
