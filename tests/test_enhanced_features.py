"""
Comprehensive test suite for enhanced document portal features.
Tests document loading, caching, evaluation, memory management, and API endpoints.
"""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application modules
from api.main import app
from utils.enhanced_document_loaders import EnhancedDocumentLoader, load_documents_enhanced
from utils.caching import CacheManager, DocumentCache, EmbeddingCache
from utils.token_counter import TokenCounter, get_token_counter
from utils.evaluation import RAGEvaluator, evaluate_response
from utils.memory_manager import MemoryManager, SessionManager, ConversationMemory
from model.models import (
    DocumentType, ProcessingStatus, DocumentMetadata, ChatSession, 
    TokenUsage, EvaluationMetrics, ChatRequest, ChatResponse
)
from exception.custom_exception import DocumentPortalException

class TestEnhancedDocumentLoaders:
    """Test enhanced document loading capabilities."""
    
    def setup_method(self):
        """Setup test environment."""
        self.loader = EnhancedDocumentLoader(extract_images=True, extract_tables=True)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_supported_extensions(self):
        """Test that all required extensions are supported."""
        expected_extensions = {'.pdf', '.docx', '.txt', '.md', '.ppt', '.pptx', 
                             '.xlsx', '.xls', '.csv', '.db', '.sqlite'}
        assert self.loader.SUPPORTED_EXTENSIONS == expected_extensions
    
    def test_text_file_loading(self):
        """Test loading plain text files."""
        # Create test text file
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test document with multiple lines.\nSecond line here."
        test_file.write_text(test_content, encoding='utf-8')
        
        documents = self.loader.load_document(test_file)
        
        assert len(documents) == 1
        assert documents[0].page_content == test_content
        assert documents[0].metadata["file_type"] == "txt"
        assert documents[0].metadata["source"] == str(test_file)
    
    def test_markdown_file_loading(self):
        """Test loading Markdown files."""
        test_file = self.temp_dir / "test.md"
        test_content = "# Test Document\n\nThis is a **markdown** document with *formatting*."
        test_file.write_text(test_content, encoding='utf-8')
        
        documents = self.loader.load_document(test_file)
        
        assert len(documents) == 1
        assert documents[0].page_content == test_content
        assert documents[0].metadata["file_type"] == "markdown"
        assert "html_version" in documents[0].metadata
    
    def test_csv_file_loading(self):
        """Test loading CSV files."""
        test_file = self.temp_dir / "test.csv"
        test_data = "Name,Age,City\nJohn,25,NYC\nJane,30,LA"
        test_file.write_text(test_data, encoding='utf-8')
        
        documents = self.loader.load_document(test_file)
        
        assert len(documents) >= 1  # At least one document (plus summary)
        # Check that summary document exists
        summary_docs = [doc for doc in documents if doc.metadata.get("file_type") == "csv_summary"]
        assert len(summary_docs) == 1
        assert "rows" in summary_docs[0].metadata
        assert "columns" in summary_docs[0].metadata
    
    def test_unsupported_file_extension(self):
        """Test handling of unsupported file extensions."""
        test_file = self.temp_dir / "test.xyz"
        test_file.write_text("test content")
        
        with pytest.raises(DocumentPortalException, match="Failed to load document"):
            self.loader.load_document(test_file)
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        nonexistent_file = self.temp_dir / "nonexistent.txt"
        
        with pytest.raises(DocumentPortalException):
            self.loader.load_document(nonexistent_file)
    
    @patch('sqlite3.connect')
    def test_sqlite_loading(self, mock_connect):
        """Test SQLite database loading."""
        # Mock SQLite connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value = mock_cursor
        
        # Mock table list
        mock_cursor.fetchall.return_value = [('users',), ('products',)]
        
        # Mock schema and data
        mock_conn.execute.side_effect = [
            Mock(fetchall=lambda: [('users',), ('products',)]),  # Tables
            Mock(fetchall=lambda: [(0, 'id', 'INTEGER'), (1, 'name', 'TEXT')]),  # Schema
            Mock(fetchall=lambda: [(1, 'John'), (2, 'Jane')]),  # Sample data
            Mock(fetchall=lambda: [(0, 'id', 'INTEGER'), (1, 'price', 'REAL')]),  # Schema
            Mock(fetchall=lambda: [(1, 99.99), (2, 149.99)])  # Sample data
        ]
        
        test_file = self.temp_dir / "test.db"
        test_file.touch()  # Create empty file
        
        documents = self.loader.load_document(test_file)
        
        assert len(documents) == 2  # Two tables
        assert all(doc.metadata["file_type"] == "sqlite" for doc in documents)


class TestCachingSystem:
    """Test caching functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cache_manager = CacheManager(
            redis_url=None,  # Use only disk and memory cache for testing
            disk_cache_dir=tempfile.mkdtemp(),
            memory_cache_size=100
        )
        self.doc_cache = DocumentCache(self.cache_manager)
        self.embedding_cache = EmbeddingCache(self.cache_manager)
    
    def test_basic_cache_operations(self):
        """Test basic cache set/get operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Test set
        success = self.cache_manager.set(key, value)
        assert success
        
        # Test get
        retrieved_value = self.cache_manager.get(key)
        assert retrieved_value == value
        
        # Test delete
        success = self.cache_manager.delete(key)
        assert success
        
        # Verify deletion
        retrieved_value = self.cache_manager.get(key)
        assert retrieved_value is None
    
    def test_document_analysis_cache(self):
        """Test document analysis caching."""
        file_path = "/test/document.pdf"
        file_hash = "abc123"
        analysis_result = {
            "summary": "Test document summary",
            "metadata": {"pages": 5, "words": 1000}
        }
        
        # Test cache miss
        cached_result = self.doc_cache.get_document_analysis(file_path, file_hash)
        assert cached_result is None
        
        # Test cache set
        success = self.doc_cache.set_document_analysis(file_path, file_hash, analysis_result)
        assert success
        
        # Test cache hit
        cached_result = self.doc_cache.get_document_analysis(file_path, file_hash)
        assert cached_result == analysis_result
    
    def test_embeddings_cache(self):
        """Test embeddings caching."""
        text_hash = "text_hash_123"
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test cache miss
        cached_embeddings = self.embedding_cache.cache.get(
            self.embedding_cache.cache._generate_key("embeddings", text_hash)
        )
        assert cached_embeddings is None
        
        # Test cache set
        key = self.embedding_cache.cache._generate_key("embeddings", text_hash)
        success = self.embedding_cache.cache.set(key, embeddings)
        assert success
        
        # Test cache hit
        cached_embeddings = self.embedding_cache.cache.get(key)
        assert cached_embeddings == embeddings
    
    def test_batch_embeddings_operations(self):
        """Test batch embeddings operations."""
        embeddings_dict = {
            "hash1": [0.1, 0.2, 0.3],
            "hash2": [0.4, 0.5, 0.6],
            "hash3": [0.7, 0.8, 0.9]
        }
        
        # Set batch embeddings
        self.embedding_cache.set_batch_embeddings(embeddings_dict)
        
        # Get batch embeddings
        text_hashes = list(embeddings_dict.keys())
        results = self.embedding_cache.get_batch_embeddings(text_hashes)
        
        for text_hash in text_hashes:
            assert results[text_hash] == embeddings_dict[text_hash]


class TestTokenCounter:
    """Test token counting and cost analysis."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_log = tempfile.mktemp(suffix=".jsonl")
        self.token_counter = TokenCounter(usage_log_path=self.temp_log)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_log):
            os.unlink(self.temp_log)
    
    def test_token_counting(self):
        """Test token counting for different models."""
        test_text = "This is a test message for token counting."
        
        # Test with different models
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]
        
        for model in models:
            token_count = self.token_counter.count_tokens(test_text, model)
            assert isinstance(token_count, int)
            assert token_count > 0
    
    def test_cost_estimation(self):
        """Test cost estimation."""
        prompt_tokens = 100
        completion_tokens = 50
        model_name = "gpt-3.5-turbo"
        
        cost = self.token_counter.estimate_cost(prompt_tokens, completion_tokens, model_name)
        
        assert isinstance(cost, float)
        assert cost > 0
        
        # Verify cost calculation
        pricing = self.token_counter.PRICING[model_name]
        expected_cost = (prompt_tokens / 1000) * pricing["input"] + (completion_tokens / 1000) * pricing["output"]
        assert abs(cost - expected_cost) < 0.0001
    
    def test_usage_record_creation(self):
        """Test creation of usage records."""
        prompt_text = "What is the capital of France?"
        completion_text = "The capital of France is Paris."
        model_name = "gpt-3.5-turbo"
        operation_type = "chat"
        session_id = "test_session_123"
        
        usage = self.token_counter.create_usage_record(
            prompt_text, completion_text, model_name, operation_type, session_id
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.model_name == model_name
        assert usage.operation_type == operation_type
        assert usage.session_id == session_id
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.estimated_cost > 0
    
    def test_usage_statistics(self):
        """Test usage statistics calculation."""
        # Create multiple usage records
        for i in range(5):
            self.token_counter.create_usage_record(
                f"Test prompt {i}",
                f"Test response {i}",
                "gpt-3.5-turbo",
                "test",
                f"session_{i % 2}"  # Two different sessions
            )
        
        stats = self.token_counter.get_usage_stats()
        
        assert stats["operations"] == 5
        assert stats["total_tokens"] > 0
        assert stats["total_cost"] > 0
        assert len(stats["by_session"]) == 2  # Two sessions
        assert "gpt-3.5-turbo" in stats["by_model"]


class TestEvaluationSystem:
    """Test evaluation system with DeepEval."""
    
    def setup_method(self):
        """Setup test environment."""
        # Mock the DeepEval components to avoid external API calls
        self.mock_evaluator = Mock(spec=RAGEvaluator)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('utils.evaluation.FaithfulnessMetric')
    @patch('utils.evaluation.AnswerRelevancyMetric')
    def test_rag_evaluator_initialization(self, mock_relevancy, mock_faithfulness):
        """Test RAG evaluator initialization."""
        model_name = "gpt-3.5-turbo"
        evaluator = RAGEvaluator(model_name)
        
        assert evaluator.model_name == model_name
        mock_faithfulness.assert_called_once()
        mock_relevancy.assert_called_once()
    
    def test_evaluation_metrics_model(self):
        """Test evaluation metrics Pydantic model."""
        metrics = EvaluationMetrics(
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.80,
            context_recall=0.75,
            harmfulness=0.05,
            bias=0.10,
            toxicity=0.02,
            overall_score=0.82,
            model_used="gpt-3.5-turbo"
        )
        
        assert metrics.faithfulness == 0.85
        assert metrics.overall_score == 0.82
        assert isinstance(metrics.evaluation_timestamp, datetime)
    
    @patch('utils.evaluation.RAGEvaluator')
    def test_evaluate_response_function(self, mock_evaluator_class):
        """Test the convenience evaluate_response function."""
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        expected_metrics = EvaluationMetrics(
            overall_score=0.85,
            model_used="gpt-3.5-turbo"
        )
        mock_evaluator.evaluate_comprehensive.return_value = expected_metrics
        
        result = evaluate_response(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            context=["AI stands for artificial intelligence."],
            session_id="test_session"
        )
        
        assert result == expected_metrics
        mock_evaluator.evaluate_comprehensive.assert_called_once()


class TestMemoryManagement:
    """Test memory and session management."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(storage_dir=self.temp_dir)
        self.memory_manager = MemoryManager()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_session_creation(self):
        """Test chat session creation."""
        user_id = "test_user_123"
        session = self.session_manager.create_session(user_id)
        
        assert isinstance(session, ChatSession)
        assert session.user_id == user_id
        assert len(session.session_id) > 0
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_message_addition(self):
        """Test adding messages to session."""
        session = self.session_manager.create_session()
        session_id = session.session_id
        
        # Add user message
        success = self.session_manager.add_message(
            session_id, "user", "Hello, how are you?", {"source": "test"}
        )
        assert success
        
        # Add assistant message
        success = self.session_manager.add_message(
            session_id, "assistant", "I'm doing well, thank you!"
        )
        assert success
        
        # Verify messages
        updated_session = self.session_manager.get_session(session_id)
        assert len(updated_session.messages) == 2
        assert updated_session.messages[0].role == "user"
        assert updated_session.messages[1].role == "assistant"
    
    def test_conversation_history_retrieval(self):
        """Test retrieving conversation history."""
        session = self.session_manager.create_session()
        session_id = session.session_id
        
        # Add multiple messages
        for i in range(10):
            self.session_manager.add_message(session_id, "user", f"Message {i}")
            self.session_manager.add_message(session_id, "assistant", f"Response {i}")
        
        # Test limited history
        history = self.session_manager.get_conversation_history(session_id, limit=5)
        assert len(history) == 5
        
        # Test full history
        full_history = self.session_manager.get_conversation_history(session_id, limit=0)
        assert len(full_history) == 20
    
    def test_session_persistence(self):
        """Test session persistence to disk."""
        session = self.session_manager.create_session("test_user")
        session_id = session.session_id
        
        # Add some messages
        self.session_manager.add_message(session_id, "user", "Test message")
        self.session_manager.add_message(session_id, "assistant", "Test response")
        
        # Create new session manager (simulating restart)
        new_session_manager = SessionManager(storage_dir=self.temp_dir)
        
        # Load session
        loaded_session = new_session_manager.get_session(session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == session_id
        assert loaded_session.user_id == "test_user"
        assert len(loaded_session.messages) == 2


class TestAPIEndpoints:
    """Test enhanced API endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "document-portal"
    
    def test_analyze_endpoint_validation(self):
        """Test document analysis endpoint validation."""
        # Test without file
        response = self.client.post("/analyze")
        assert response.status_code == 422  # Validation error
    
    def test_chat_query_validation(self):
        """Test chat query endpoint validation."""
        # Test without required parameters
        response = self.client.post("/chat/query")
        assert response.status_code == 422  # Validation error
        
        # Test with missing session when required
        response = self.client.post(
            "/chat/query",
            data={
                "question": "Test question",
                "use_session_dirs": True
                # Missing session_id
            }
        )
        assert response.status_code == 400  # Bad request


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""
    
    def test_document_metadata_model(self):
        """Test DocumentMetadata model."""
        metadata = DocumentMetadata(
            file_path="/test/document.pdf",
            file_name="document.pdf",
            file_type=DocumentType.PDF,
            file_size=1024,
            page_count=5,
            has_tables=True,
            has_images=False
        )
        
        assert metadata.file_type == DocumentType.PDF
        assert metadata.status == ProcessingStatus.PENDING
        assert isinstance(metadata.created_at, datetime)
        
        # Test serialization
        data = metadata.model_dump()
        assert data["file_type"] == "pdf"  # Enum value
        assert "created_at" in data
    
    def test_chat_request_validation(self):
        """Test ChatRequest model validation."""
        # Valid request
        request = ChatRequest(
            question="What is AI?",
            session_id="test_session",
            k=5
        )
        assert request.question == "What is AI?"
        assert request.k == 5
        
        # Test validation errors
        with pytest.raises(ValueError):
            ChatRequest(question="")  # Empty question
        
        with pytest.raises(ValueError):
            ChatRequest(question="x" * 2001)  # Too long question
        
        with pytest.raises(ValueError):
            ChatRequest(question="Valid question", k=0)  # Invalid k value
    
    def test_token_usage_model(self):
        """Test TokenUsage model."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.001,
            model_name="gpt-3.5-turbo",
            operation_type="chat"
        )
        
        assert usage.total_tokens == 150
        assert usage.estimated_cost == 0.001
        assert isinstance(usage.timestamp, datetime)


class TestErrorHandling:
    """Test error handling and exception management."""
    
    def test_document_portal_exception(self):
        """Test custom exception handling."""
        original_error = ValueError("Original error")
        
        with pytest.raises(DocumentPortalException) as exc_info:
            raise DocumentPortalException("Custom error message", original_error)
        
        assert "Custom error message" in str(exc_info.value)
    
    def test_file_not_found_handling(self):
        """Test handling of file not found errors."""
        loader = EnhancedDocumentLoader()
        
        with pytest.raises(DocumentPortalException):
            loader.load_document("/nonexistent/file.pdf")


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.client = TestClient(app)
    
    def teardown_method(self):
        """Cleanup integration test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_document_processing_workflow(self):
        """Test complete document processing workflow."""
        # Create test document
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test document for integration testing."
        test_file.write_text(test_content)
        
        # Test document loading
        loader = EnhancedDocumentLoader()
        documents = loader.load_document(test_file)
        
        assert len(documents) == 1
        assert documents[0].page_content == test_content
        
        # Test caching
        cache_manager = CacheManager(redis_url=None, disk_cache_dir=str(self.temp_dir / "cache"))
        doc_cache = DocumentCache(cache_manager)
        
        file_hash = "test_hash_123"
        analysis_result = {"summary": "Test summary", "metadata": {"words": 10}}
        
        success = doc_cache.set_document_analysis(str(test_file), file_hash, analysis_result)
        assert success
        
        cached_result = doc_cache.get_document_analysis(str(test_file), file_hash)
        assert cached_result == analysis_result
    
    @patch('utils.token_counter.get_token_counter')
    def test_token_tracking_workflow(self, mock_get_counter):
        """Test token tracking in complete workflow."""
        mock_counter = Mock()
        mock_counter.create_usage_record.return_value = TokenUsage(
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            estimated_cost=0.0001,
            model_name="gpt-3.5-turbo",
            operation_type="test"
        )
        mock_get_counter.return_value = mock_counter
        
        # Simulate API call with token tracking
        from utils.token_counter import log_llm_usage
        
        usage = log_llm_usage(
            prompt="Test prompt",
            response="Test response",
            model_name="gpt-3.5-turbo",
            operation_type="test"
        )
        
        assert usage.total_tokens == 75
        assert usage.estimated_cost == 0.0001


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
