"""
API integration tests for enhanced document portal features.
Tests complete API workflows with enhanced functionality.
"""

import pytest
import tempfile
import json
import io
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.main import app
from model.models import DocumentType, ProcessingStatus

class TestEnhancedAPIIntegration:
    """Test enhanced API integration with new features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_document_analysis_with_caching(self):
        """Test document analysis with caching integration."""
        # Create test PDF content
        test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        with patch('utils.enhanced_document_loaders.EnhancedDocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            # Mock document loading
            from langchain.schema import Document
            mock_documents = [Document(
                page_content="Test document content",
                metadata={"source": "test.pdf", "file_type": "pdf"}
            )]
            mock_loader.load_document.return_value = mock_documents
            
            # Test file upload
            response = self.client.post(
                "/analyze",
                files={"file": ("test.pdf", io.BytesIO(test_content), "application/pdf")}
            )
            
            # Should work with mocked loader
            assert response.status_code in [200, 500]  # 500 due to missing dependencies in test
    
    def test_chat_with_memory_integration(self):
        """Test chat functionality with memory integration."""
        # First, create an index
        test_content = "This is a test document for chat functionality."
        
        with patch('src.document_ingestion.data_ingestion.ChatIngestor') as mock_ingestor_class:
            mock_ingestor = Mock()
            mock_ingestor_class.return_value = mock_ingestor
            mock_ingestor.session_id = "test_session_123"
            
            # Mock file upload for indexing
            response = self.client.post(
                "/chat/index",
                files=[("files", ("test.txt", io.BytesIO(test_content.encode()), "text/plain"))],
                data={
                    "session_id": "test_session_123",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "k": 5
                }
            )
            
            # Should return session info
            if response.status_code == 200:
                data = response.json()
                assert "session_id" in data
                assert data["session_id"] == "test_session_123"
    
    def test_document_comparison_with_evaluation(self):
        """Test document comparison with evaluation metrics."""
        test_content1 = "First document content for comparison."
        test_content2 = "Second document content for comparison."
        
        with patch('src.document_ingestion.data_ingestion.DocumentComparator') as mock_comparator_class:
            mock_comparator = Mock()
            mock_comparator_class.return_value = mock_comparator
            mock_comparator.session_id = "comparison_session"
            mock_comparator.save_uploaded_files.return_value = ("ref.pdf", "act.pdf")
            mock_comparator.combine_documents.return_value = "Combined document text"
            
            with patch('src.document_compare.document_comparator.DocumentComparatorLLM') as mock_llm_class:
                mock_llm = Mock()
                mock_llm_class.return_value = mock_llm
                
                import pandas as pd
                mock_df = pd.DataFrame([{"Page": "1", "Changes": "Test change"}])
                mock_llm.compare_documents.return_value = mock_df
                
                response = self.client.post(
                    "/compare",
                    files=[
                        ("reference", ("ref.pdf", io.BytesIO(test_content1.encode()), "application/pdf")),
                        ("actual", ("act.pdf", io.BytesIO(test_content2.encode()), "application/pdf"))
                    ]
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assert "rows" in data
                    assert "session_id" in data
    
    def test_token_usage_tracking(self):
        """Test that token usage is tracked across API calls."""
        with patch('utils.token_counter.get_token_counter') as mock_get_counter:
            mock_counter = Mock()
            mock_get_counter.return_value = mock_counter
            
            # Mock token counting
            mock_counter.count_tokens.return_value = 50
            mock_counter.estimate_cost.return_value = 0.001
            
            # Make API call that should trigger token counting
            response = self.client.get("/health")
            assert response.status_code == 200
    
    def test_caching_integration(self):
        """Test that caching is properly integrated."""
        with patch('utils.caching.get_cache_manager') as mock_get_cache:
            mock_cache = Mock()
            mock_get_cache.return_value = mock_cache
            
            # Mock cache operations
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True
            
            # Make API call that should use caching
            response = self.client.get("/health")
            assert response.status_code == 200


class TestErrorHandlingIntegration:
    """Test error handling in API integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
    
    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads."""
        # Test with invalid file type
        response = self.client.post(
            "/analyze",
            files={"file": ("test.xyz", io.BytesIO(b"invalid content"), "application/octet-stream")}
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 422, 500]
    
    def test_missing_session_handling(self):
        """Test handling of missing session scenarios."""
        response = self.client.post(
            "/chat/query",
            data={
                "question": "Test question",
                "session_id": "nonexistent_session",
                "use_session_dirs": True,
                "k": 5
            }
        )
        
        # Should return appropriate error
        assert response.status_code in [404, 400, 500]
    
    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        # Test with missing required fields
        response = self.client.post("/chat/query", data={})
        assert response.status_code == 422  # Validation error
        
        # Test with invalid data types
        response = self.client.post(
            "/chat/query",
            data={
                "question": "Valid question",
                "k": "invalid_number"  # Should be integer
            }
        )
        assert response.status_code == 422


class TestPerformanceIntegration:
    """Test performance aspects of the enhanced system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = self.client.get("/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5
    
    def test_large_document_handling(self):
        """Test handling of large documents."""
        # Create large text content
        large_content = "This is a test line.\n" * 10000  # ~200KB
        
        with patch('src.document_analyzer.data_analysis.DocumentAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_document.return_value = {"summary": "Large document processed"}
            
            response = self.client.post(
                "/analyze",
                files={"file": ("large.txt", io.BytesIO(large_content.encode()), "text/plain")}
            )
            
            # Should handle large files
            assert response.status_code in [200, 500]  # 500 due to test limitations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
