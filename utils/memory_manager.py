"""
Enhanced memory and chat history management using LangChain memory components.
Supports conversation history, document context, and session management.
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage as CoreBaseMessage

from model.models import ChatSession, ChatMessage, DocumentMetadata
from utils.caching import get_cache_manager
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

class SessionManager:
    """
    Manages chat sessions with persistent storage and memory.
    """
    
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.cache = get_cache_manager()
        self.active_sessions: Dict[str, ChatSession] = {}
        
    def create_session(self, user_id: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            user_id=user_id
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        log.info(f"Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID, loading from storage if needed."""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from storage
        session = self._load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
            return session
        
        return None
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            log.warning(f"Session not found: {session_id}")
            return False
        
        session.add_message(role, content, metadata)
        self._save_session(session)
        
        # Cache recent messages for quick access
        cache_key = f"session_messages:{session_id}"
        recent_messages = session.messages[-10:]  # Keep last 10 messages in cache
        self.cache.set(cache_key, recent_messages, ttl=3600)
        
        return True
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return session.messages[-limit:] if limit > 0 else session.messages
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session context including documents and metadata."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.messages),
            "document_ids": session.document_ids,
            "user_id": session.user_id
        }
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up sessions older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                session = self._load_session_from_file(session_file)
                if session and session.last_activity < cutoff_date:
                    session_file.unlink()
                    if session.session_id in self.active_sessions:
                        del self.active_sessions[session.session_id]
                    log.info(f"Cleaned up old session: {session.session_id}")
            except Exception as e:
                log.warning(f"Failed to cleanup session {session_file}: {e}")
    
    def _save_session(self, session: ChatSession):
        """Save session to storage."""
        try:
            session_file = self.storage_dir / f"{session.session_id}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.model_dump(), f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save session {session.session_id}: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from storage."""
        session_file = self.storage_dir / f"{session_id}.json"
        return self._load_session_from_file(session_file)
    
    def _load_session_from_file(self, session_file: Path) -> Optional[ChatSession]:
        """Load session from file."""
        try:
            if not session_file.exists():
                return None
            
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            if isinstance(data.get("created_at"), str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("last_activity"), str):
                data["last_activity"] = datetime.fromisoformat(data["last_activity"])
            
            # Convert message timestamps
            for msg in data.get("messages", []):
                if isinstance(msg.get("timestamp"), str):
                    msg["timestamp"] = datetime.fromisoformat(msg["timestamp"])
            
            return ChatSession(**data)
            
        except Exception as e:
            log.error(f"Failed to load session from {session_file}: {e}")
            return None


class ConversationMemory:
    """
    Enhanced conversation memory using LangChain memory components.
    """
    
    def __init__(self, session_id: str, memory_type: str = "buffer", max_token_limit: int = 2000):
        self.session_id = session_id
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        
        # Initialize LangChain memory
        history_file = f"data/chat_histories/{session_id}.json"
        Path(history_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.chat_history = FileChatMessageHistory(file_path=history_file)
        
        if memory_type == "summary":
            llm = ModelLoader().load_llm()
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                chat_memory=self.chat_history,
                max_token_limit=max_token_limit,
                return_messages=True
            )
        else:
            self.memory = ConversationBufferMemory(
                chat_memory=self.chat_history,
                return_messages=True
            )
    
    def add_user_message(self, message: str):
        """Add user message to memory."""
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str):
        """Add AI message to memory."""
        self.memory.chat_memory.add_ai_message(message)
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for prompt formatting."""
        return self.memory.load_memory_variables({})
    
    def get_conversation_string(self) -> str:
        """Get conversation as formatted string."""
        memory_vars = self.get_memory_variables()
        messages = memory_vars.get("history", [])
        
        if isinstance(messages, list):
            conversation_parts = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_parts.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation_parts.append(f"Assistant: {msg.content}")
            return "\n".join(conversation_parts)
        
        return str(messages)
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_message_count(self) -> int:
        """Get number of messages in memory."""
        return len(self.memory.chat_memory.messages)


class DocumentContextManager:
    """
    Manages document context and relationships within sessions.
    """
    
    def __init__(self):
        self.cache = get_cache_manager()
    
    def add_document_to_session(self, session_id: str, document_metadata: DocumentMetadata):
        """Add document to session context."""
        cache_key = f"session_docs:{session_id}"
        
        # Get existing documents
        existing_docs = self.cache.get(cache_key) or []
        
        # Add new document if not already present
        doc_dict = document_metadata.dict()
        if not any(doc["file_path"] == doc_dict["file_path"] for doc in existing_docs):
            existing_docs.append(doc_dict)
            self.cache.set(cache_key, existing_docs, ttl=86400)  # 24 hours
            
            log.info(f"Added document to session {session_id}: {document_metadata.file_name}")
    
    def get_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all documents associated with a session."""
        cache_key = f"session_docs:{session_id}"
        return self.cache.get(cache_key) or []
    
    def remove_document_from_session(self, session_id: str, file_path: str):
        """Remove document from session context."""
        cache_key = f"session_docs:{session_id}"
        existing_docs = self.cache.get(cache_key) or []
        
        updated_docs = [doc for doc in existing_docs if doc["file_path"] != file_path]
        self.cache.set(cache_key, updated_docs, ttl=86400)
        
        log.info(f"Removed document from session {session_id}: {file_path}")
    
    def get_document_relationships(self, session_id: str) -> Dict[str, Any]:
        """Analyze relationships between documents in a session."""
        documents = self.get_session_documents(session_id)
        
        if not documents:
            return {"documents": [], "relationships": []}
        
        # Group by file type
        by_type = {}
        for doc in documents:
            file_type = doc.get("file_type", "unknown")
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append(doc)
        
        # Identify potential relationships
        relationships = []
        
        # Same type documents
        for file_type, docs in by_type.items():
            if len(docs) > 1:
                relationships.append({
                    "type": "same_format",
                    "description": f"Multiple {file_type} documents",
                    "documents": [doc["file_name"] for doc in docs]
                })
        
        # Documents with similar names
        doc_names = [doc["file_name"] for doc in documents]
        for i, name1 in enumerate(doc_names):
            for j, name2 in enumerate(doc_names[i+1:], i+1):
                # Simple similarity check
                common_words = set(name1.lower().split()) & set(name2.lower().split())
                if len(common_words) > 0:
                    relationships.append({
                        "type": "similar_names",
                        "description": f"Documents with similar names: {', '.join(common_words)}",
                        "documents": [name1, name2]
                    })
        
        return {
            "documents": documents,
            "by_type": by_type,
            "relationships": relationships,
            "total_documents": len(documents)
        }


class MemoryManager:
    """
    Central memory manager coordinating all memory components.
    """
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.document_context = DocumentContextManager()
        self.conversation_memories: Dict[str, ConversationMemory] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session:
                return session
        
        return self.session_manager.create_session(user_id)
    
    def get_conversation_memory(self, session_id: str, memory_type: str = "buffer") -> ConversationMemory:
        """Get conversation memory for session."""
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = ConversationMemory(session_id, memory_type)
        
        return self.conversation_memories[session_id]
    
    def add_exchange(self, session_id: str, user_message: str, ai_response: str, metadata: Optional[Dict] = None):
        """Add complete exchange to both session and memory."""
        # Add to session
        self.session_manager.add_message(session_id, "user", user_message, metadata)
        self.session_manager.add_message(session_id, "assistant", ai_response, metadata)
        
        # Add to conversation memory
        conv_memory = self.get_conversation_memory(session_id)
        conv_memory.add_user_message(user_message)
        conv_memory.add_ai_message(ai_response)
    
    def get_context_for_query(self, session_id: str, include_documents: bool = True) -> Dict[str, Any]:
        """Get comprehensive context for a query."""
        context = {
            "session": self.session_manager.get_session_context(session_id),
            "conversation": self.get_conversation_memory(session_id).get_memory_variables(),
            "message_count": self.get_conversation_memory(session_id).get_message_count()
        }
        
        if include_documents:
            context["documents"] = self.document_context.get_document_relationships(session_id)
        
        return context
    
    def cleanup_inactive_sessions(self, days: int = 7):
        """Clean up inactive sessions and memories."""
        self.session_manager.cleanup_old_sessions(days)
        
        # Clean up conversation memories for non-existent sessions
        active_session_ids = set(self.session_manager.active_sessions.keys())
        memory_session_ids = set(self.conversation_memories.keys())
        
        for session_id in memory_session_ids - active_session_ids:
            if session_id in self.conversation_memories:
                del self.conversation_memories[session_id]
                log.info(f"Cleaned up conversation memory for session: {session_id}")


# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def get_session(session_id: Optional[str] = None, user_id: Optional[str] = None) -> ChatSession:
    """Convenience function to get or create session."""
    return get_memory_manager().get_or_create_session(session_id, user_id)

def add_conversation_exchange(session_id: str, user_message: str, ai_response: str, metadata: Optional[Dict] = None):
    """Convenience function to add conversation exchange."""
    get_memory_manager().add_exchange(session_id, user_message, ai_response, metadata)
