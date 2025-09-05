"""
Token counter and cost analysis utility for API usage monitoring.
Supports multiple LLM providers with accurate token counting and cost estimation.
"""

import os
import json
import tiktoken
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from model.models import TokenUsage
from logger import GLOBAL_LOGGER as log

class TokenCounter:
    """
    Token counter with cost analysis for multiple LLM providers.
    """
    
    # Pricing per 1K tokens (Free models have $0 cost)
    PRICING = {
        # Free models
        "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
        "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
        "gemini-pro": {"input": 0.0, "output": 0.0},
        "deepseek-r1-distill-llama-70b": {"input": 0.0, "output": 0.0},
        "llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
        "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
        "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
        "gemma-7b-it": {"input": 0.0, "output": 0.0},
        # Legacy paid models (kept for reference)
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    # Model name mappings for tiktoken (free models use approximate counting)
    TIKTOKEN_MODELS = {
        # Free models - use gpt-3.5-turbo for token counting approximation
        "gemini-2.0-flash": "gpt-3.5-turbo",
        "gemini-1.5-flash": "gpt-3.5-turbo",
        "gemini-pro": "gpt-3.5-turbo",
        "deepseek-r1-distill-llama-70b": "gpt-3.5-turbo",
        "llama-3.1-70b-versatile": "gpt-3.5-turbo",
        "llama-3.1-8b-instant": "gpt-3.5-turbo",
        "mixtral-8x7b-32768": "gpt-3.5-turbo",
        "gemma-7b-it": "gpt-3.5-turbo",
        # Legacy paid models
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "claude-3-opus": "gpt-4",
        "claude-3-sonnet": "gpt-4",
        "claude-3-haiku": "gpt-3.5-turbo"
    }
    
    def __init__(self, usage_log_path: str = "logs/token_usage.jsonl"):
        self.usage_log_path = Path(usage_log_path)
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders cache
        self._encoders = {}
    
    def get_encoder(self, model_name: str) -> tiktoken.Encoding:
        """Get tiktoken encoder for model."""
        if model_name not in self._encoders:
            tiktoken_model = self.TIKTOKEN_MODELS.get(model_name, "gpt-3.5-turbo")
            try:
                self._encoders[model_name] = tiktoken.encoding_for_model(tiktoken_model)
            except KeyError:
                # Fallback to cl100k_base encoding
                self._encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoders[model_name]
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for specific model."""
        try:
            encoder = self.get_encoder(model_name)
            return len(encoder.encode(text))
        except Exception as e:
            log.warning(f"Token counting failed for {model_name}: {e}")
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def count_tokens_batch(self, texts: List[str], model_name: str) -> List[int]:
        """Count tokens for multiple texts."""
        encoder = self.get_encoder(model_name)
        return [len(encoder.encode(text)) for text in texts]
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """Estimate cost based on token usage."""
        if model_name not in self.PRICING:
            log.warning(f"No pricing info for {model_name}, using gpt-3.5-turbo rates")
            model_name = "gpt-3.5-turbo"
        
        pricing = self.PRICING[model_name]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def create_usage_record(
        self,
        prompt_text: str,
        completion_text: str,
        model_name: str,
        operation_type: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TokenUsage:
        """Create a complete token usage record."""
        
        prompt_tokens = self.count_tokens(prompt_text, model_name)
        completion_tokens = self.count_tokens(completion_text, model_name)
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = self.estimate_cost(prompt_tokens, completion_tokens, model_name)
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            model_name=model_name,
            session_id=session_id,
            operation_type=operation_type
        )
        
        # Log usage
        self.log_usage(usage, metadata)
        
        return usage
    
    def log_usage(self, usage: TokenUsage, metadata: Optional[Dict[str, Any]] = None):
        """Log token usage to file."""
        try:
            log_entry = {
                "timestamp": usage.timestamp.isoformat(),
                "model_name": usage.model_name,
                "operation_type": usage.operation_type,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost": usage.estimated_cost,
                "session_id": usage.session_id,
                "metadata": metadata or {}
            }
            
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            log.error(f"Failed to log token usage: {e}")
    
    def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage statistics with filtering."""
        
        if not self.usage_log_path.exists():
            return {"total_cost": 0, "total_tokens": 0, "operations": 0}
        
        stats = {
            "total_cost": 0.0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "operations": 0,
            "by_model": {},
            "by_operation": {},
            "by_session": {},
            "daily_usage": {}
        }
        
        try:
            with open(self.usage_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_date = datetime.fromisoformat(entry["timestamp"])
                        
                        # Apply filters
                        if start_date and entry_date < start_date:
                            continue
                        if end_date and entry_date > end_date:
                            continue
                        if model_name and entry["model_name"] != model_name:
                            continue
                        if operation_type and entry["operation_type"] != operation_type:
                            continue
                        if session_id and entry.get("session_id") != session_id:
                            continue
                        
                        # Aggregate stats
                        stats["total_cost"] += entry["estimated_cost"]
                        stats["total_tokens"] += entry["total_tokens"]
                        stats["prompt_tokens"] += entry["prompt_tokens"]
                        stats["completion_tokens"] += entry["completion_tokens"]
                        stats["operations"] += 1
                        
                        # By model
                        model = entry["model_name"]
                        if model not in stats["by_model"]:
                            stats["by_model"][model] = {"cost": 0, "tokens": 0, "operations": 0}
                        stats["by_model"][model]["cost"] += entry["estimated_cost"]
                        stats["by_model"][model]["tokens"] += entry["total_tokens"]
                        stats["by_model"][model]["operations"] += 1
                        
                        # By operation
                        op = entry["operation_type"]
                        if op not in stats["by_operation"]:
                            stats["by_operation"][op] = {"cost": 0, "tokens": 0, "operations": 0}
                        stats["by_operation"][op]["cost"] += entry["estimated_cost"]
                        stats["by_operation"][op]["tokens"] += entry["total_tokens"]
                        stats["by_operation"][op]["operations"] += 1
                        
                        # By session
                        sess = entry.get("session_id", "unknown")
                        if sess not in stats["by_session"]:
                            stats["by_session"][sess] = {"cost": 0, "tokens": 0, "operations": 0}
                        stats["by_session"][sess]["cost"] += entry["estimated_cost"]
                        stats["by_session"][sess]["tokens"] += entry["total_tokens"]
                        stats["by_session"][sess]["operations"] += 1
                        
                        # Daily usage
                        day = entry_date.date().isoformat()
                        if day not in stats["daily_usage"]:
                            stats["daily_usage"][day] = {"cost": 0, "tokens": 0, "operations": 0}
                        stats["daily_usage"][day]["cost"] += entry["estimated_cost"]
                        stats["daily_usage"][day]["tokens"] += entry["total_tokens"]
                        stats["daily_usage"][day]["operations"] += 1
                        
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            log.error(f"Failed to read usage stats: {e}")
        
        return stats
    
    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed cost breakdown for the last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = self.get_usage_stats(start_date=start_date, end_date=end_date)
        
        # Calculate averages
        if stats["operations"] > 0:
            stats["avg_cost_per_operation"] = stats["total_cost"] / stats["operations"]
            stats["avg_tokens_per_operation"] = stats["total_tokens"] / stats["operations"]
        else:
            stats["avg_cost_per_operation"] = 0
            stats["avg_tokens_per_operation"] = 0
        
        # Calculate daily averages
        if days > 0:
            stats["avg_daily_cost"] = stats["total_cost"] / days
            stats["avg_daily_tokens"] = stats["total_tokens"] / days
            stats["avg_daily_operations"] = stats["operations"] / days
        
        return stats
    
    def export_usage_report(self, output_path: str, days: int = 30) -> bool:
        """Export detailed usage report to JSON file."""
        try:
            report = {
                "report_generated": datetime.now().isoformat(),
                "period_days": days,
                "cost_breakdown": self.get_cost_breakdown(days),
                "pricing_info": self.PRICING
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            log.info(f"Usage report exported to: {output_path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to export usage report: {e}")
            return False
    
    def set_budget_alert(self, daily_limit: float, monthly_limit: float) -> Dict[str, Any]:
        """Check if usage exceeds budget limits."""
        today_stats = self.get_usage_stats(
            start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
        
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_stats = self.get_usage_stats(start_date=month_start)
        
        alerts = {
            "daily_usage": today_stats["total_cost"],
            "daily_limit": daily_limit,
            "daily_exceeded": today_stats["total_cost"] > daily_limit,
            "monthly_usage": monthly_stats["total_cost"],
            "monthly_limit": monthly_limit,
            "monthly_exceeded": monthly_stats["total_cost"] > monthly_limit,
            "daily_percentage": (today_stats["total_cost"] / daily_limit * 100) if daily_limit > 0 else 0,
            "monthly_percentage": (monthly_stats["total_cost"] / monthly_limit * 100) if monthly_limit > 0 else 0
        }
        
        if alerts["daily_exceeded"] or alerts["monthly_exceeded"]:
            log.warning("Budget limits exceeded", alerts=alerts)
        
        return alerts


# Global token counter instance
_token_counter = None

def get_token_counter() -> TokenCounter:
    """Get global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count_tokens(text, model_name)

def estimate_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-3.5-turbo") -> float:
    """Convenience function to estimate cost."""
    return get_token_counter().estimate_cost(prompt_tokens, completion_tokens, model_name)

def log_llm_usage(
    prompt: str,
    response: str,
    model_name: str,
    operation_type: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TokenUsage:
    """Convenience function to log LLM usage."""
    return get_token_counter().create_usage_record(
        prompt, response, model_name, operation_type, session_id, metadata
    )
