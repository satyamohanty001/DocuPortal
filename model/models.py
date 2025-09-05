from pydantic import BaseModel, RootModel, Field, field_validator
from typing import List, Union, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str

class ChangeFormat(BaseModel):
    Page: str
    Changes: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class PromptType(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"

# Enhanced Pydantic Models for Industry-Ready Application

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    POWERPOINT = "powerpoint"
    EXCEL = "excel"
    CSV = "csv"
    SQLITE = "sqlite"
    DATABASE = "database"
    MONGODB = "mongodb"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(BaseModel):
    """Enhanced document metadata with comprehensive information."""
    file_path: str
    file_name: str
    file_type: DocumentType
    file_size: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Content metadata
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    
    # Processing flags
    has_tables: bool = False
    has_images: bool = False
    has_ocr_content: bool = False
    
    # Additional metadata
    language: Optional[str] = None
    encoding: Optional[str] = None
    checksum: Optional[str] = None
    
    class Config:
        use_enum_values = True

class ChatMessage(BaseModel):
    """Chat message model for conversation history."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = Field(..., description="Session identifier")
    metadata: Optional[Dict[str, Any]] = None

class ChatSession(BaseModel):
    """Chat session model for managing conversations."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    messages: List[ChatMessage] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            session_id=self.session_id,
            metadata=metadata
        )
        self.messages.append(message)
        self.last_activity = datetime.now()

class TokenUsage(BaseModel):
    """Token usage tracking for cost analysis."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    operation_type: str  # 'analysis', 'comparison', 'chat', etc.

class CacheEntry(BaseModel):
    """Cache entry model for storing processed results."""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.now)
    size_bytes: Optional[int] = None

class EvaluationMetrics(BaseModel):
    """Evaluation metrics for DeepEval integration."""
    faithfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_relevancy: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    harmfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias: Optional[float] = Field(None, ge=0.0, le=1.0)
    toxicity: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str
    session_id: Optional[str] = None

class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis."""
    file_path: Optional[str] = None
    file_content: Optional[bytes] = None
    extract_tables: bool = True
    extract_images: bool = True
    perform_ocr: bool = True
    language: Optional[str] = "en"
    
    @field_validator('file_path', 'file_content')
    @classmethod
    def validate_input(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        if not values.get('file_path') and not values.get('file_content'):
            raise ValueError('Either file_path or file_content must be provided')
        return v

class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis."""
    document_id: str
    metadata: DocumentMetadata
    analysis_results: Dict[str, Any]
    token_usage: TokenUsage
    processing_time: float
    status: ProcessingStatus
    error_message: Optional[str] = None

class DocumentComparisonRequest(BaseModel):
    """Request model for document comparison."""
    reference_document: str = Field(..., description="Reference document path/ID")
    target_document: str = Field(..., description="Target document path/ID")
    comparison_type: str = Field(default="semantic", description="Type of comparison")
    include_metadata: bool = True

class DocumentComparisonResponse(BaseModel):
    """Response model for document comparison."""
    comparison_id: str
    reference_doc_id: str
    target_doc_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    differences: List[ChangeFormat]
    token_usage: TokenUsage
    processing_time: float
    status: ProcessingStatus

class ChatRequest(BaseModel):
    """Request model for chat queries."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    use_session_dirs: bool = True
    k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True
    stream_response: bool = False

class ChatResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str
    token_usage: TokenUsage
    evaluation_metrics: Optional[EvaluationMetrics] = None
    processing_time: float
    cached: bool = False

class SystemHealth(BaseModel):
    """System health monitoring model."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, str] = Field(default_factory=dict)
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    active_sessions: int = 0
    cache_hit_rate: Optional[float] = None

class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    processing_time: Optional[float] = None