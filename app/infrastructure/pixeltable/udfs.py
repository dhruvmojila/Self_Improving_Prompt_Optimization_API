"""
Metric User Defined Functions (UDFs) for Pixeltable.
Defines evaluation metrics as Pixeltable UDFs for computed columns.
"""

import pixeltable as pxt
from typing import Optional, Dict, Any
import json
import re
import logging
from difflib import SequenceMatcher

# For semantic similarity (using simple approach for now)
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Tier 1: Deterministic Metrics (Fast, Free)
# ============================================================================

@pxt.udf
def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact string match metric.
    Returns 1.0 if strings match exactly (case-insensitive), else 0.0.
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        1.0 if match, 0.0 otherwise
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


@pxt.udf
def substring_match(prediction: str, ground_truth: str) -> float:
    """
    Check if ground truth is a substring of prediction.
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        1.0 if ground_truth in prediction, 0.0 otherwise
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    return 1.0 if ground_truth.lower() in prediction.lower() else 0.0


@pxt.udf
def regex_match(prediction: str, pattern: str) -> float:
    """
    Check if prediction matches regex pattern.
    
    Args:
        prediction: Model prediction
        pattern: Regex pattern
        
    Returns:
        1.0 if matches, 0.0 otherwise
    """
    if prediction is None or pattern is None:
        return 0.0
    
    try:
        return 1.0 if re.search(pattern, prediction, re.IGNORECASE) else 0.0
    except Exception:
        return 0.0


@pxt.udf
def json_schema_valid(prediction: str) -> float:
    """
    Check if prediction is valid JSON.
    
    Args:
        prediction: Model prediction
        
    Returns:
        1.0 if valid JSON, 0.0 otherwise
    """
    if prediction is None:
        return 0.0
    
    try:
        json.loads(prediction)
        return 1.0
    except Exception:
        return 0.0


# ============================================================================
# Tier 2: Heuristic Metrics (Fast, Cheap)
# ============================================================================

@pxt.udf
def string_similarity(prediction: str, ground_truth: str) -> float:
    """
    String similarity using SequenceMatcher (like difflib).
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        Similarity score 0.0-1.0
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    matcher = SequenceMatcher(None, prediction.lower(), ground_truth.lower())
    return matcher.ratio()


@pxt.udf
def word_overlap(prediction: str, ground_truth: str) -> float:
    """
    Word overlap score (Jaccard similarity).
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        Jaccard similarity of word sets
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    pred_words = set(prediction.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
    
    intersection = pred_words & truth_words
    union = pred_words | truth_words
    
    return len(intersection) / len(union) if union else 0.0


@pxt.udf
def length_ratio(prediction: str, ground_truth: str) -> float:
    """
    Ratio of prediction length to ground truth length.
    Useful for detecting overly verbose or brief outputs.
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        Length ratio (1.0 = same length, >1 = longer, <1 = shorter)
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    truth_len = len(ground_truth.strip())
    if truth_len == 0:
        return 1.0
    
    pred_len = len(prediction.strip())
    return pred_len / truth_len


# ============================================================================
# Composite Metrics
# ============================================================================

@pxt.udf
def correctness_metric(prediction: str, ground_truth: str) -> float:
    """
    Combined correctness metric.
    
    Averages exact match, substring match, and string similarity.
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        
    Returns:
        Correctness score 0.0-1.0
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    # Exact match (high weight)
    exact = exact_match(prediction, ground_truth)
    if exact == 1.0:
        return 1.0
    
    # Substring match (medium weight)
    substr = substring_match(prediction, ground_truth)
    
    # String similarity (for partial credit)
    similarity = string_similarity(prediction, ground_truth)
    
    # Weighted average
    return 0.5 * substr + 0.5 * similarity


@pxt.udf
def quality_metric(
    prediction: str,
    ground_truth: str,
    correctness_weight: float = 0.7,
    conciseness_weight: float = 0.3
) -> float:
    """
    Quality metric combining correctness and conciseness.
    
    Args:
        prediction: Model prediction
        ground_truth: Expected output
        correctness_weight: Weight for correctness (0-1)
        conciseness_weight: Weight for conciseness (0-1)
        
    Returns:
        Quality score 0.0-1.0
    """
    if prediction is None or ground_truth is None:
        return 0.0
    
    # Correctness
    correct_score = correctness_metric(prediction, ground_truth)
    
    # Conciseness (penalize overly long outputs)
    len_ratio = length_ratio(prediction, ground_truth)
    # Ideal ratio is 1.0, penalize deviation
    conciseness_score = 1.0 - min(abs(len_ratio - 1.0), 1.0)
    
    # Weighted combination
    return (correctness_weight * correct_score + 
            conciseness_weight * conciseness_score)


# ============================================================================
# Helper Functions for UDF Registration
# ============================================================================

def register_all_udfs():
    """
    Register all metric UDFs with Pixeltable.
    Call this during initialization.
    """
    logger.info("Registering metric UDFs with Pixeltable...")
    
    # UDFs are automatically registered when decorated with @pxt.udf
    # This function just logs for confirmation
    
    registered_udfs = [
        'exact_match',
        'substring_match',
        'regex_match',
        'json_schema_valid',
        'string_similarity',
        'word_overlap',
        'length_ratio',
        'correctness_metric',
        'quality_metric',
    ]
    
    logger.info(f"Registered {len(registered_udfs)} metric UDFs")
    return registered_udfs


def get_metric_function(metric_name: str):
    """
    Get metric function by name.
    
    Args:
        metric_name: Name of metric
        
    Returns:
        Metric UDF function
    """
    metrics = {
        'exact_match': exact_match,
        'substring_match': substring_match,
        'regex_match': regex_match,
        'json_schema_valid': json_schema_valid,
        'string_similarity': string_similarity,
        'word_overlap': word_overlap,
        'length_ratio': length_ratio,
        'correctness_metric': correctness_metric,
        'quality_metric': quality_metric,
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return metrics[metric_name]


# ============================================================================
# Metric Metadata
# ============================================================================

METRIC_INFO = {
    'exact_match': {
        'description': 'Exact string match (case-insensitive)',
        'tier': 1,
        'cost': 'free',
        'latency': '<1ms',
        'range': '0.0-1.0',
    },
    'substring_match': {
        'description': 'Ground truth is substring of prediction',
        'tier': 1,
        'cost': 'free',
        'latency': '<1ms',
        'range': '0.0-1.0',
    },
    'regex_match': {
        'description': 'Prediction matches regex pattern',
        'tier': 1,
        'cost': 'free',
        'latency': '<1ms',
        'range': '0.0-1.0',
    },
    'json_schema_valid': {
        'description': 'Prediction is valid JSON',
        'tier': 1,
        'cost': 'free',
        'latency': '<1ms',
        'range': '0.0-1.0',
    },
    'string_similarity': {
        'description': 'SequenceMatcher string similarity',
        'tier': 2,
        'cost': 'free',
        'latency': '<10ms',
        'range': '0.0-1.0',
    },
    'word_overlap': {
        'description': 'Jaccard similarity of word sets',
        'tier': 2,
        'cost': 'free',
        'latency': '<10ms',
        'range': '0.0-1.0',
    },
    'correctness_metric': {
        'description': 'Composite: exact + substring + similarity',
        'tier': 2,
        'cost': 'free',
        'latency': '<10ms',
        'range': '0.0-1.0',
    },
    'quality_metric': {
        'description': 'Composite: correctness + conciseness',
        'tier': 2,
        'cost': 'free',
        'latency': '<10ms',
        'range': '0.0-1.0',
    },
}
