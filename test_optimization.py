"""
End-to-end optimization test script.
Demonstrates the complete optimization pipeline with DSPy MIPRO.
"""

import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.domain.services import DSPyOptimizer, create_metric_function, EvaluatorService, LineageService
from app.domain.models import OptimizationConfig, OptimizerType
from app.infrastructure.pixeltable import init_pixeltable, init_tables
from app.core.events import event_bus

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_optimization_demo():
    """
    Run a complete optimization demo.
    
    Steps:
    1. Initialize Pixeltable
    2. Create sample dataset
    3. Create baseline prompt
    4. Run MIPRO optimization
    5. Show results with lineage
    """
    
    logger.info("=" * 60)
    logger.info("DSPy Optimization Demo - Sentiment Analysis")
    logger.info("=" * 60)
    
    # Step 1: Initialize
    logger.info("\n1. Initializing Pixeltable...")
    init_pixeltable()
    init_tables()
    logger.info("✓ Pixeltable ready")
    
    # Step 2: Create sample dataset
    logger.info("\n2. Creating sample sentiment dataset...")
    dataset_id = await create_sample_dataset()
    logger.info(f"✓ Dataset created: {dataset_id}")
    
    # Step 3: Create baseline prompt
    logger.info("\n3. Creating baseline prompt...")
    prompt_id = await create_baseline_prompt()
    logger.info(f"✓ Baseline prompt created: {prompt_id}")
    
    # Step 4: Run optimization
    logger.info("\n4. Running MIPRO optimization...")
    logger.info("   This will:")
    logger.info("   - Use teacher model (Groq llama-3.3-70b) to generate reasoning traces")
    logger.info("   - Search instruction space with Bayesian optimization")
    logger.info("   - Select optimal few-shot examples")
    logger.info("   - Evaluate on dev set using correctness metric")
    
    config = OptimizationConfig(
        teacher_model="groq/llama-3.3-70b-versatile",
        student_model="groq/llama-3.1-8b-instant",
        teacher_provider="groq",
        student_provider="groq",
        optimizer_type=OptimizerType.MIPRO,
        num_trials=5,  # Reduced for demo
        num_fewshot_examples=3,
        budget_usd=1.0,
        metric_name="correctness_metric"
    )
    
    optimizer = DSPyOptimizer(config)
    
    def progress_callback(pct, msg):
        logger.info(f"   [{pct:3.0f}%] {msg}")
    
    metric_fn = create_metric_function("correctness_metric")
    
    try:
        result = optimizer.optimize_mipro(
            prompt_id=prompt_id,
            dataset_id=dataset_id,
            metric_fn=metric_fn,
            progress_callback=progress_callback
        )
        
        logger.info("\n✅ Optimization Complete!")
        logger.info(f"   Baseline score: {result['baseline_score']:.3f}")
        logger.info(f"   Optimized score: {result['optimized_score']:.3f}")
        logger.info(f"   Improvement: {result['improvement_pct']:+.1f}%")
        logger.info(f"   Optimizer: {result['optimizer_type']}")
        logger.info(f"   Few-shot examples: {result['num_examples']}")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Show lineage
    logger.info("\n5. Prompt lineage tracking...")
    lineage_service = LineageService()
    
    try:
        # Create optimized version in registry
        optimized_prompt_id = await create_optimized_prompt(prompt_id, result)
        
        # Get lineage
        lineage = lineage_service.get_lineage(optimized_prompt_id)
        logger.info(f"✓ Lineage tracked: {lineage.total_versions} versions")
        
        # Compute diff
        diff = lineage_service.compute_diff(prompt_id, optimized_prompt_id)
        logger.info(f"✓ Semantic diff computed")
        if diff.metric_delta_pct:
            logger.info(f"   Performance delta: {diff.metric_delta_pct:+.1f}%")
        
    except Exception as e:
        logger.warning(f"⚠️  Lineage tracking skipped: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)


async def create_sample_dataset() -> str:
    """Create a sample sentiment analysis dataset."""
    from app.infrastructure.pixeltable import get_schemas
    import uuid
    from datetime import datetime
    
    dataset_id = str(uuid.uuid4())
    
    # Sample data
    samples = [
        # Train set
        {"text": "I love this product!", "sentiment": "positive", "split": "train"},
        {"text": "This is terrible quality.", "sentiment": "negative", "split": "train"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral", "split": "train"},
        {"text": "Best purchase ever!", "sentiment": "positive", "split": "train"},
        {"text": "Complete waste of money.", "sentiment": "negative", "split": "train"},
        {"text": "Decent for the price.", "sentiment": "neutral", "split": "train"},
        
        # Dev set
        {"text": "Amazing quality and fast shipping!", "sentiment": "positive", "split": "dev"},
        {"text": "Broke after one use.", "sentiment": "negative", "split": "dev"},
        {"text": "Average product.", "sentiment": "neutral", "split": "dev"},
    ]
    
    schemas = get_schemas()
    table = schemas.get_table('datasets')
    
    for idx, sample in enumerate(samples):
        table.insert([{
            'dataset_id': dataset_id,
            'row_id': f"{dataset_id}_{idx}",
            'input_data': {"text": sample["text"]},
            'ground_truth': {"sentiment": sample["sentiment"]},
            'split': sample["split"],
            'modality': 'text',
            'dataset_name': 'Sentiment Analysis Demo',
            'project_id': '',
            'created_at': datetime.utcnow(),
        }])
    
    return dataset_id


async def create_baseline_prompt() -> str:
    """Create a baseline sentiment analysis prompt."""
    from app.domain.services import LineageService
    
    lineage_service = LineageService()
    
    prompt_data = {
        'name': 'Baseline Sentiment Classifier',
        'template': 'Classify the sentiment of this text: {text}',
        'signature': {
            'inputs': ['text'],
            'outputs': ['sentiment']
        },
        'description': 'Simple sentiment classification prompt',
        'tags': ['baseline'],
        'author_id': 'demo_user'
    }
    
    version = lineage_service.create_version(prompt_data)
    return version.id


async def create_optimized_prompt(parent_id: str, optimization_result: dict) -> str:
    """Create optimized prompt version."""
    from app.domain.services import LineageService
    
    lineage_service = LineageService()
    
    prompt_data = {
        'name': f'Optimized Sentiment Classifier (MIPRO)',
        'template': 'Classify the sentiment of this text: {text}\n\nSentiment:',
        'signature': {
            'inputs': ['text'],
            'outputs': ['sentiment']
        },
        'description': f"MIPRO-optimized with {optimization_result['improvement_pct']:+.1f}% improvement",
        'tags': ['optimized', 'mipro'],
        'author_id': 'dspy_optimizer',
        'artifact_path': optimization_result['artifact_path'],
        'optimizer_type': optimization_result['optimizer_type'],
        'baseline_metric': optimization_result['baseline_score'],
        'optimized_metric': optimization_result['optimized_score'],
    }
    
    version = lineage_service.create_version(prompt_data, parent_id=parent_id)
    return version.id


if __name__ == "__main__":
    asyncio.run(run_optimization_demo())
