"""
DeepEval integration for comprehensive RAG evaluation metrics.
Supports faithfulness, answer relevancy, context precision, and other metrics.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase

from model.models import EvaluationMetrics, ChatMessage
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

class RAGEvaluator:
    """
    Comprehensive RAG evaluation using DeepEval metrics.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Initialize metrics
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=0.7,
            model=model_name,
            include_reason=True
        )
        
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=model_name,
            include_reason=True
        )
        
        self.contextual_precision_metric = ContextualPrecisionMetric(
            threshold=0.7,
            model=model_name,
            include_reason=True
        )
        
        self.contextual_recall_metric = ContextualRecallMetric(
            threshold=0.7,
            model=model_name,
            include_reason=True
        )
        
        self.hallucination_metric = HallucinationMetric(
            threshold=0.5,
            model=model_name,
            include_reason=True
        )
        
        self.bias_metric = BiasMetric(
            threshold=0.5,
            model=model_name,
            include_reason=True
        )
        
        self.toxicity_metric = ToxicityMetric(
            threshold=0.5,
            model=model_name,
            include_reason=True
        )
        
        log.info(f"RAGEvaluator initialized with model: {model_name}")
    
    def create_test_case(
        self,
        input_query: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        retrieval_context: Optional[List[str]] = None
    ) -> LLMTestCase:
        """Create a test case for evaluation."""
        return LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context or []
        )
    
    def evaluate_faithfulness(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate faithfulness of the response to the context."""
        try:
            self.faithfulness_metric.measure(test_case)
            return {
                "score": self.faithfulness_metric.score,
                "success": self.faithfulness_metric.success,
                "reason": self.faithfulness_metric.reason,
                "metric": "faithfulness"
            }
        except Exception as e:
            log.error(f"Faithfulness evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "faithfulness"}
    
    def evaluate_answer_relevancy(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate relevancy of the answer to the question."""
        try:
            self.answer_relevancy_metric.measure(test_case)
            return {
                "score": self.answer_relevancy_metric.score,
                "success": self.answer_relevancy_metric.success,
                "reason": self.answer_relevancy_metric.reason,
                "metric": "answer_relevancy"
            }
        except Exception as e:
            log.error(f"Answer relevancy evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "answer_relevancy"}
    
    def evaluate_context_precision(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate precision of the retrieved context."""
        try:
            self.contextual_precision_metric.measure(test_case)
            return {
                "score": self.contextual_precision_metric.score,
                "success": self.contextual_precision_metric.success,
                "reason": self.contextual_precision_metric.reason,
                "metric": "context_precision"
            }
        except Exception as e:
            log.error(f"Context precision evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "context_precision"}
    
    def evaluate_context_recall(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate recall of the retrieved context."""
        try:
            self.contextual_recall_metric.measure(test_case)
            return {
                "score": self.contextual_recall_metric.score,
                "success": self.contextual_recall_metric.success,
                "reason": self.contextual_recall_metric.reason,
                "metric": "context_recall"
            }
        except Exception as e:
            log.error(f"Context recall evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "context_recall"}
    
    def evaluate_hallucination(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate hallucination in the response."""
        try:
            self.hallucination_metric.measure(test_case)
            return {
                "score": 1.0 - self.hallucination_metric.score,  # Invert score (lower hallucination = better)
                "success": self.hallucination_metric.success,
                "reason": self.hallucination_metric.reason,
                "metric": "hallucination"
            }
        except Exception as e:
            log.error(f"Hallucination evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "hallucination"}
    
    def evaluate_bias(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate bias in the response."""
        try:
            self.bias_metric.measure(test_case)
            return {
                "score": 1.0 - self.bias_metric.score,  # Invert score (lower bias = better)
                "success": self.bias_metric.success,
                "reason": self.bias_metric.reason,
                "metric": "bias"
            }
        except Exception as e:
            log.error(f"Bias evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "bias"}
    
    def evaluate_toxicity(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Evaluate toxicity in the response."""
        try:
            self.toxicity_metric.measure(test_case)
            return {
                "score": 1.0 - self.toxicity_metric.score,  # Invert score (lower toxicity = better)
                "success": self.toxicity_metric.success,
                "reason": self.toxicity_metric.reason,
                "metric": "toxicity"
            }
        except Exception as e:
            log.error(f"Toxicity evaluation failed: {e}")
            return {"score": 0.0, "success": False, "reason": str(e), "metric": "toxicity"}
    
    def evaluate_comprehensive(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: List[str],
        expected_output: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> EvaluationMetrics:
        """Run comprehensive evaluation with all metrics."""
        
        test_case = self.create_test_case(
            input_query=input_query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        # Run all evaluations
        results = {}
        
        try:
            results["faithfulness"] = self.evaluate_faithfulness(test_case)
            results["answer_relevancy"] = self.evaluate_answer_relevancy(test_case)
            results["context_precision"] = self.evaluate_context_precision(test_case)
            results["context_recall"] = self.evaluate_context_recall(test_case)
            results["hallucination"] = self.evaluate_hallucination(test_case)
            results["bias"] = self.evaluate_bias(test_case)
            results["toxicity"] = self.evaluate_toxicity(test_case)
            
            # Calculate overall score
            scores = [r["score"] for r in results.values() if r["score"] is not None]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Create evaluation metrics object
            metrics = EvaluationMetrics(
                faithfulness=results["faithfulness"]["score"],
                answer_relevancy=results["answer_relevancy"]["score"],
                context_precision=results["context_precision"]["score"],
                context_recall=results["context_recall"]["score"],
                harmfulness=1.0 - results["hallucination"]["score"],  # Convert back
                bias=1.0 - results["bias"]["score"],  # Convert back
                toxicity=1.0 - results["toxicity"]["score"],  # Convert back
                overall_score=overall_score,
                model_used=self.model_name,
                session_id=session_id
            )
            
            log.info(f"Comprehensive evaluation completed. Overall score: {overall_score:.3f}")
            return metrics
            
        except Exception as e:
            log.error(f"Comprehensive evaluation failed: {e}")
            raise DocumentPortalException(f"Evaluation failed: {str(e)}", e)
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[EvaluationMetrics]:
        """Evaluate multiple test cases in batch."""
        results = []
        
        for i, case in enumerate(test_cases):
            try:
                log.info(f"Evaluating test case {i+1}/{len(test_cases)}")
                
                metrics = self.evaluate_comprehensive(
                    input_query=case["input_query"],
                    actual_output=case["actual_output"],
                    retrieval_context=case.get("retrieval_context", []),
                    expected_output=case.get("expected_output"),
                    session_id=session_id
                )
                
                results.append(metrics)
                
            except Exception as e:
                log.error(f"Failed to evaluate test case {i+1}: {e}")
                # Create a failed metrics object
                failed_metrics = EvaluationMetrics(
                    overall_score=0.0,
                    model_used=self.model_name,
                    session_id=session_id
                )
                results.append(failed_metrics)
        
        return results
    
    def generate_evaluation_report(
        self,
        metrics_list: List[EvaluationMetrics],
        output_path: str
    ) -> bool:
        """Generate a comprehensive evaluation report."""
        try:
            import json
            
            # Calculate aggregate statistics
            total_metrics = len(metrics_list)
            if total_metrics == 0:
                return False
            
            # Aggregate scores
            aggregate = {
                "total_evaluations": total_metrics,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "aggregate_scores": {},
                "individual_results": []
            }
            
            # Calculate averages for each metric
            metrics_fields = [
                "faithfulness", "answer_relevancy", "context_precision", 
                "context_recall", "harmfulness", "bias", "toxicity", "overall_score"
            ]
            
            for field in metrics_fields:
                scores = [getattr(m, field) for m in metrics_list if getattr(m, field) is not None]
                if scores:
                    aggregate["aggregate_scores"][field] = {
                        "average": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "count": len(scores)
                    }
            
            # Add individual results
            for metrics in metrics_list:
                result = {
                    "faithfulness": metrics.faithfulness,
                    "answer_relevancy": metrics.answer_relevancy,
                    "context_precision": metrics.context_precision,
                    "context_recall": metrics.context_recall,
                    "harmfulness": metrics.harmfulness,
                    "bias": metrics.bias,
                    "toxicity": metrics.toxicity,
                    "overall_score": metrics.overall_score,
                    "timestamp": metrics.evaluation_timestamp.isoformat(),
                    "session_id": metrics.session_id
                }
                aggregate["individual_results"].append(result)
            
            # Write report
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=2, ensure_ascii=False)
            
            log.info(f"Evaluation report generated: {output_path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to generate evaluation report: {e}")
            return False


class ConversationEvaluator:
    """Evaluate entire conversations for quality and coherence."""
    
    def __init__(self, rag_evaluator: RAGEvaluator):
        self.rag_evaluator = rag_evaluator
    
    def evaluate_conversation(
        self,
        messages: List[ChatMessage],
        retrieval_contexts: List[List[str]],
        session_id: str
    ) -> Dict[str, Any]:
        """Evaluate an entire conversation."""
        
        if len(messages) % 2 != 0:
            log.warning("Conversation has odd number of messages, skipping last message")
            messages = messages[:-1]
        
        conversation_metrics = []
        
        # Evaluate each Q&A pair
        for i in range(0, len(messages), 2):
            if i + 1 >= len(messages):
                break
                
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            if user_msg.role != "user" or assistant_msg.role != "assistant":
                continue
            
            context = retrieval_contexts[i // 2] if i // 2 < len(retrieval_contexts) else []
            
            try:
                metrics = self.rag_evaluator.evaluate_comprehensive(
                    input_query=user_msg.content,
                    actual_output=assistant_msg.content,
                    retrieval_context=context,
                    session_id=session_id
                )
                conversation_metrics.append(metrics)
                
            except Exception as e:
                log.error(f"Failed to evaluate message pair {i//2 + 1}: {e}")
        
        # Calculate conversation-level statistics
        if not conversation_metrics:
            return {"error": "No valid message pairs to evaluate"}
        
        # Aggregate metrics
        fields = ["faithfulness", "answer_relevancy", "context_precision", 
                 "context_recall", "harmfulness", "bias", "toxicity", "overall_score"]
        
        conversation_stats = {
            "session_id": session_id,
            "total_exchanges": len(conversation_metrics),
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for field in fields:
            scores = [getattr(m, field) for m in conversation_metrics if getattr(m, field) is not None]
            if scores:
                conversation_stats["metrics"][field] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "trend": self._calculate_trend(scores)
                }
        
        return conversation_stats
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction for scores."""
        if len(scores) < 2:
            return "stable"
        
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"


# Global evaluator instances
_rag_evaluator = None
_conversation_evaluator = None

def get_rag_evaluator(model_name: str = "gpt-3.5-turbo") -> RAGEvaluator:
    """Get global RAG evaluator instance."""
    global _rag_evaluator
    if _rag_evaluator is None or _rag_evaluator.model_name != model_name:
        _rag_evaluator = RAGEvaluator(model_name)
    return _rag_evaluator

def get_conversation_evaluator(model_name: str = "gpt-3.5-turbo") -> ConversationEvaluator:
    """Get global conversation evaluator instance."""
    global _conversation_evaluator
    if _conversation_evaluator is None:
        rag_eval = get_rag_evaluator(model_name)
        _conversation_evaluator = ConversationEvaluator(rag_eval)
    return _conversation_evaluator

def evaluate_response(
    question: str,
    answer: str,
    context: List[str],
    session_id: Optional[str] = None,
    model_name: str = "gpt-3.5-turbo"
) -> EvaluationMetrics:
    """Convenience function to evaluate a single response."""
    evaluator = get_rag_evaluator(model_name)
    return evaluator.evaluate_comprehensive(
        input_query=question,
        actual_output=answer,
        retrieval_context=context,
        session_id=session_id
    )
