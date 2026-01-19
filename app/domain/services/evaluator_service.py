"""
Evaluator Service - Multi-tier evaluation pipeline.
Implements deterministic, heuristic, and LLM-as-a-Judge evaluation strategies.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from app.infrastructure.llm import LLMFactory, LLMConfig
from app.infrastructure.pixeltable.udfs import (
    exact_match, substring_match, string_similarity,
    word_overlap, correctness_metric, quality_metric, METRIC_INFO
)

logger = logging.getLogger(__name__)


class EvaluationTier(str, Enum):
    """Evaluation tier levels."""
    TIER_1_DETERMINISTIC = "tier1"  # Fast, free (regex, exact match)
    TIER_2_HEURISTIC = "tier2"  # Fast, cheap (BLEU, ROUGE, similarity)
    TIER_3_LLM_JUDGE = "tier3"  # Slow, expensive (LLM evaluation)


class EvaluatorService:
    """
    Multi-tier evaluation service.
    
    Provides three tiers of evaluation:
    - Tier 1: Deterministic (regex, JSON schema, exact match)
    - Tier 2: Heuristic (BLEU, ROUGE, string similarity)
    - Tier 3: LLM-as-a-Judge (nuanced quality assessment)
    """
    
    def __init__(self):
        self.judge_model = None
    
    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metric_name: str = "correctness_metric",
        tier: EvaluationTier = EvaluationTier.TIER_2_HEURISTIC
    ) -> float:
        """
        Evaluate a prediction against ground truth.
        
        Args:
            prediction: Model output
            ground_truth: Expected output
            metric_name: Metric to use
            tier: Evaluation tier
            
        Returns:
            Score (0.0 to 1.0)
        """
        if tier == EvaluationTier.TIER_1_DETERMINISTIC:
            return self._evaluate_tier1(prediction, ground_truth, metric_name)
        elif tier == EvaluationTier.TIER_2_HEURISTIC:
            return self._evaluate_tier2(prediction, ground_truth, metric_name)
        elif tier == EvaluationTier.TIER_3_LLM_JUDGE:
            return self._evaluate_tier3(prediction, ground_truth, metric_name)
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    def _evaluate_tier1(self, prediction: str, ground_truth: str, metric_name: str) -> float:
        """Tier 1: Deterministic evaluation."""
        if metric_name == "exact_match":
            return float(exact_match(prediction, ground_truth))
        elif metric_name == "substring_match":
            return float(substring_match(prediction, ground_truth))
        else:
            # Default to exact match for tier 1
            return float(exact_match(prediction, ground_truth))
    
    def _evaluate_tier2(self, prediction: str, ground_truth: str, metric_name: str) -> float:
        """Tier 2: Heuristic evaluation."""
        if metric_name == "string_similarity":
            return float(string_similarity(prediction, ground_truth))
        elif metric_name == "word_overlap":
            return float(word_overlap(prediction, ground_truth))
        elif metric_name == "correctness_metric":
            return float(correctness_metric(prediction, ground_truth))
        elif metric_name == "quality_metric":
            return float(quality_metric(prediction, ground_truth))
        else:
            # Default to correctness metric
            return float(correctness_metric(prediction, ground_truth))
    
    def _evaluate_tier3(self, prediction: str, ground_truth: str, metric_name: str) -> float:
        """Tier 3: LLM-as-a-Judge evaluation."""
        if self.judge_model is None:
            self.judge_model = LLMFactory.create_judge()
        
        # Create judge prompt
        judge_prompt = self._create_judge_prompt(prediction, ground_truth, metric_name)
        
        # Get judgment
        config = LLMConfig(temperature=0.1, max_tokens=200)
        
        try:
            response = self.judge_model.generate(
                prompt=judge_prompt,
                config=config
            )
            
            # Parse score from response
            score = self._parse_judge_score(response.content)
            return score
            
        except Exception as e:
            logger.error(f"LLM-as-a-Judge failed: {e}")
            # Fallback to tier 2
            return self._evaluate_tier2(prediction, ground_truth, metric_name)
    
    def _create_judge_prompt(self, prediction: str, ground_truth: str, metric_name: str) -> str:
        """Create prompt for LLM-as-a-Judge."""
        if metric_name == "correctness_metric":
            prompt = f"""You are an AI judge evaluating the correctness of a model's answer.

Ground Truth Answer: {ground_truth}

Model's Answer: {prediction}

Rate the correctness of the model's answer on a scale from 0.0 to 1.0:
- 1.0 = Completely correct, semantically equivalent to ground truth
- 0.8 = Mostly correct, minor errors or different phrasing
- 0.5 = Partially correct, contains some correct information
- 0.2 = Mostly incorrect, major errors
- 0.0 = Completely wrong or nonsensical

Provide only the numeric score (e.g., 0.85) without explanation."""

        elif metric_name == "quality_metric":
            prompt = f"""You are an AI judge evaluating the quality of a model's answer.

Expected Answer: {ground_truth}

Model's Answer: {prediction}

Rate the overall quality considering correctness, conciseness, and clarity on a scale from 0.0 to 1.0:
- 1.0 = Perfect answer - correct, concise, clear
- 0.8 = Good answer - correct with minor verbosity or clarity issues
- 0.5 = Acceptable answer - correct but verbose or unclear
- 0.2 = Poor answer - incorrect or very unclear
- 0.0 = Unacceptable answer

Provide only the numeric score (e.g., 0.75) without explanation."""

        else:
            # Generic rubric
            prompt = f"""You are an AI judge evaluating a model's answer.

Expected: {ground_truth}
Actual: {prediction}

Rate on a scale from 0.0 to 1.0. Provide only the numeric score."""
        
        return prompt
    
    def _parse_judge_score(self, response: str) -> float:
        """Parse numeric score from judge response."""
        import re
        
        # Look for float pattern
        match = re.search(r'(\d+\.?\d*)', response.strip())
        
        if match:
            score = float(match.group(1))
            # Normalize if needed
            if score > 1.0:
                score = score / 100.0  # Handle percentage format
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        # Fallback: check for qualitative ratings
        response_lower = response.lower()
        if "perfect" in response_lower or "excellent" in response_lower:
            return 1.0
        elif "good" in response_lower:
            return 0.8
        elif "acceptable" in response_lower or "ok" in response_lower:
            return 0.5
        elif "poor" in response_lower:
            return 0.2
        else:
            return 0.0
    
    def batch_evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        metric_name: str = "correctness_metric",
        tier: EvaluationTier = EvaluationTier.TIER_2_HEURISTIC
    ) -> List[float]:
        """
        Batch evaluate multiple predictions.
        
        Args:
            predictions: List of model outputs
            ground_truths: List of expected outputs
            metric_name: Metric to use
            tier: Evaluation tier
            
        Returns:
            List of scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            score = self.evaluate(pred, truth, metric_name, tier)
            scores.append(score)
        
        return scores
    
    def evaluate_with_breakdown(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate with multiple metrics and return breakdown.
        
        Args:
            prediction: Model output
            ground_truth: Expected output
            
        Returns:
            Dictionary of metric scores
        """
        breakdown = {
            'exact_match': float(exact_match(prediction, ground_truth)),
            'substring_match': float(substring_match(prediction, ground_truth)),
            'string_similarity': float(string_similarity(prediction, ground_truth)),
            'word_overlap': float(word_overlap(prediction, ground_truth)),
            'correctness_metric': float(correctness_metric(prediction, ground_truth)),
            'quality_metric': float(quality_metric(prediction, ground_truth)),
        }
        
        return breakdown
    
    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """Get metadata about a metric."""
        return METRIC_INFO.get(metric_name, {
            'description': 'Unknown metric',
            'tier': 'unknown',
            'cost': 'unknown',
            'latency': 'unknown',
        })
    
    def list_available_metrics(self) -> List[str]:
        """List all available metrics."""
        return list(METRIC_INFO.keys())


class PairwiseComparator:
    """
    Pairwise comparison evaluator.
    
    Uses LLM-as-a-Judge to compare two outputs side-by-side.
    More stable than absolute scoring.
    """
    
    def __init__(self):
        self.judge_model = LLMFactory.create_judge()
    
    def compare(
        self,
        prediction_a: str,
        prediction_b: str,
        ground_truth: Optional[str] = None,
        criteria: str = "overall quality"
    ) -> Dict[str, Any]:
        """
        Compare two predictions.
        
        Args:
            prediction_a: First prediction
            prediction_b: Second prediction
            ground_truth: Optional expected answer
            criteria: Comparison criteria
            
        Returns:
            Comparison result with winner and reasoning
        """
        prompt = self._create_comparison_prompt(
            prediction_a, prediction_b, ground_truth, criteria
        )
        
        config = LLMConfig(temperature=0.1, max_tokens=300)
        
        response = self.judge_model.generate(prompt=prompt, config=config)
        
        # Parse result
        result = self._parse_comparison_result(response.content)
        
        return result
    
    def _create_comparison_prompt(
        self,
        prediction_a: str,
        prediction_b: str,
        ground_truth: Optional[str],
        criteria: str
    ) -> str:
        """Create pairwise comparison prompt."""
        prompt = f"""You are an AI judge comparing two model outputs.

{"Expected Answer: " + ground_truth if ground_truth else ""}

Output A: {prediction_a}

Output B: {prediction_b}

Compare these outputs based on {criteria}.

Respond in this format:
Winner: [A or B or TIE]
Reasoning: [Brief explanation]
"""
        return prompt
    
    def _parse_comparison_result(self, response: str) -> Dict[str, Any]:
        """Parse comparison result."""
        import re
        
        # Extract winner
        winner_match = re.search(r'Winner:\s*(A|B|TIE)', response, re.IGNORECASE)
        winner = winner_match.group(1).upper() if winner_match else "TIE"
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        return {
            'winner': winner,
            'reasoning': reasoning,
            'raw_response': response
        }
