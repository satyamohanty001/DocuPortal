"""
Comprehensive caching system for document portal with Redis and in-memory support.
Supports embeddings, responses, and document processing caching.
"""

import json
import hashlib
import pickle
import time
import os
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from pathlib import Path

import redis
import diskcache
from langchain.schema import Document

from logger import GLOBAL_LOGGER as log
from model.models import CacheEntry, TokenUsage
from exception.custom_exception import DocumentPortalException

class CacheManager:
    """
    Multi-tier caching system supporting Redis, disk cache, and in-memory cache.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        disk_cache_dir: str = "cache",
        memory_cache_size: int = 1000,
        default_ttl: int = 3600  # 1 hour
    ):
        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_size = memory_cache_size
        
        # Initialize Redis cache
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                log.info("Redis cache initialized successfully")
            except Exception as e:
                log.warning(f"Redis initialization failed: {e}, falling back to disk cache")
                self.redis_client = None
        
        # Initialize disk cache
        try:
            self.disk_cache = diskcache.Cache(disk_cache_dir)
            log.info(f"Disk cache initialized at: {disk_cache_dir}")
        except Exception as e:
            log.error(f"Disk cache initialization failed: {e}")
            self.disk_cache = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    def _cleanup_memory_cache(self):
        """Remove expired entries and maintain size limit."""
        now = datetime.now()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry.expires_at and entry.expires_at < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Maintain size limit (LRU eviction)
        if len(self.memory_cache) > self.memory_cache_size:
            # Sort by last_accessed and remove oldest
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            items_to_remove = len(self.memory_cache) - self.memory_cache_size
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self.memory_cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries memory -> Redis -> disk)."""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.expires_at or entry.expires_at > datetime.now():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    log.debug(f"Cache hit (memory): {key}")
                    return entry.value
                else:
                    del self.memory_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        value = pickle.loads(cached_data)
                        # Store in memory cache for faster access
                        self._store_in_memory(key, value)
                        log.debug(f"Cache hit (Redis): {key}")
                        return value
                except Exception as e:
                    log.warning(f"Redis get failed for {key}: {e}")
            
            # Try disk cache
            if self.disk_cache:
                try:
                    value = self.disk_cache.get(key)
                    if value is not None:
                        # Store in memory cache for faster access
                        self._store_in_memory(key, value)
                        log.debug(f"Cache hit (disk): {key}")
                        return value
                except Exception as e:
                    log.warning(f"Disk cache get failed for {key}: {e}")
            
            log.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            log.error(f"Cache get error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (stores in all available tiers)."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        try:
            # Store in memory cache
            self._store_in_memory(key, value, expires_at)
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    serialized_value = pickle.dumps(value)
                    self.redis_client.setex(key, ttl, serialized_value)
                    log.debug(f"Stored in Redis: {key}")
                except Exception as e:
                    log.warning(f"Redis set failed for {key}: {e}")
            
            # Store in disk cache
            if self.disk_cache:
                try:
                    self.disk_cache.set(key, value, expire=ttl)
                    log.debug(f"Stored in disk cache: {key}")
                except Exception as e:
                    log.warning(f"Disk cache set failed for {key}: {e}")
            
            return True
            
        except Exception as e:
            log.error(f"Cache set error for {key}: {e}")
            return False
    
    def _store_in_memory(self, key: str, value: Any, expires_at: Optional[datetime] = None):
        """Store value in memory cache."""
        self._cleanup_memory_cache()
        
        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=len(pickle.dumps(value)) if value else 0
        )
        
        self.memory_cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        success = True
        
        # Delete from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                log.warning(f"Redis delete failed for {key}: {e}")
                success = False
        
        # Delete from disk cache
        if self.disk_cache:
            try:
                self.disk_cache.delete(key)
            except Exception as e:
                log.warning(f"Disk cache delete failed for {key}: {e}")
                success = False
        
        return success
    
    def clear(self) -> bool:
        """Clear all cache tiers."""
        success = True
        
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear Redis cache
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                log.warning(f"Redis clear failed: {e}")
                success = False
        
        # Clear disk cache
        if self.disk_cache:
            try:
                self.disk_cache.clear()
            except Exception as e:
                log.warning(f"Disk cache clear failed: {e}")
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_limit": self.memory_cache_size,
            "redis_available": self.redis_client is not None,
            "disk_cache_available": self.disk_cache is not None
        }
        
        # Redis stats
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats["redis_memory_usage"] = redis_info.get("used_memory_human", "N/A")
                stats["redis_keys"] = self.redis_client.dbsize()
            except Exception as e:
                log.warning(f"Failed to get Redis stats: {e}")
        
        # Disk cache stats
        if self.disk_cache:
            try:
                stats["disk_cache_size"] = len(self.disk_cache)
                stats["disk_cache_volume"] = self.disk_cache.volume()
            except Exception as e:
                log.warning(f"Failed to get disk cache stats: {e}")
        
        return stats


class DocumentCache:
    """Specialized cache for document processing operations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def get_document_analysis(self, file_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached document analysis results."""
        key = self.cache._generate_key("doc_analysis", f"{file_path}:{file_hash}")
        return self.cache.get(key)
    
    def set_document_analysis(self, file_path: str, file_hash: str, analysis: Dict[str, Any], ttl: int = 7200):
        """Cache document analysis results."""
        key = self.cache._generate_key("doc_analysis", f"{file_path}:{file_hash}")
        return self.cache.set(key, analysis, ttl)
    
    def get_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embeddings."""
        key = self.cache._generate_key("embeddings", text_hash)
        return self.cache.get(key)
    
    def set_embeddings(self, text_hash: str, embeddings: List[float], ttl: int = 86400):
        """Cache embeddings (24 hour TTL)."""
        key = self.cache._generate_key("embeddings", text_hash)
        return self.cache.set(key, embeddings, ttl)
    
    def get_chat_response(self, question: str, context_hash: str, session_id: str) -> Optional[str]:
        """Get cached chat response."""
        cache_key = f"{question}:{context_hash}:{session_id}"
        key = self.cache._generate_key("chat_response", cache_key)
        return self.cache.get(key)
    
    def set_chat_response(self, question: str, context_hash: str, session_id: str, response: str, ttl: int = 3600):
        """Cache chat response."""
        cache_key = f"{question}:{context_hash}:{session_id}"
        key = self.cache._generate_key("chat_response", cache_key)
        return self.cache.set(key, response, ttl)
    
    def get_document_comparison(self, doc1_hash: str, doc2_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached document comparison."""
        # Ensure consistent ordering
        hashes = sorted([doc1_hash, doc2_hash])
        key = self.cache._generate_key("doc_comparison", f"{hashes[0]}:{hashes[1]}")
        return self.cache.get(key)
    
    def set_document_comparison(self, doc1_hash: str, doc2_hash: str, comparison: Dict[str, Any], ttl: int = 7200):
        """Cache document comparison."""
        hashes = sorted([doc1_hash, doc2_hash])
        key = self.cache._generate_key("doc_comparison", f"{hashes[0]}:{hashes[1]}")
        return self.cache.set(key, comparison, ttl)


class EmbeddingCache:
    """Specialized cache for embeddings with batch operations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def get_batch_embeddings(self, text_hashes: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings in batch."""
        results = {}
        for text_hash in text_hashes:
            key = self.cache._generate_key("embeddings", text_hash)
            results[text_hash] = self.cache.get(key)
        return results
    
    def set_batch_embeddings(self, embeddings_dict: Dict[str, List[float]], ttl: int = 86400):
        """Set multiple embeddings in batch."""
        for text_hash, embeddings in embeddings_dict.items():
            key = self.cache._generate_key("embeddings", text_hash)
            self.cache.set(key, embeddings, ttl)
    
    def get_document_embeddings(self, document_id: str) -> Optional[List[List[float]]]:
        """Get all embeddings for a document."""
        key = self.cache._generate_key("doc_embeddings", document_id)
        return self.cache.get(key)
    
    def set_document_embeddings(self, document_id: str, embeddings: List[List[float]], ttl: int = 86400):
        """Cache all embeddings for a document."""
        key = self.cache._generate_key("doc_embeddings", document_id)
        return self.cache.set(key, embeddings, ttl)


# Global cache instances
_cache_manager = None
_document_cache = None
_embedding_cache = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        redis_url = os.getenv("REDIS_URL")
        _cache_manager = CacheManager(redis_url=redis_url)
    return _cache_manager

def get_document_cache() -> DocumentCache:
    """Get global document cache instance."""
    global _document_cache
    if _document_cache is None:
        _document_cache = DocumentCache(get_cache_manager())
    return _document_cache

def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(get_cache_manager())
    return _embedding_cache

def cache_key_for_file(file_path: Union[str, Path]) -> str:
    """Generate cache key for file based on path and modification time."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def cache_key_for_text(text: str) -> str:
    """Generate cache key for text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
