"""
Updated end-to-end optimization test using ProductionDSPyOptimizer.
Tests the complete pipeline with Pixeltable integration.
"""

import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.domain.services import ProductionDSPyOptimizer, create_metric_function
from app.domain.models import OptimizationConfig, OptimizerType
from app.infrastructure.pixeltable import init_pixeltable, init_tables, get_schemas
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_production_optimizer():
    """Test the production DSPy optimizer with real dataset."""
    
    logger.info("=" * 70)
    logger.info("PRODUCTION DSPy OPTIMIZER TEST")
    logger.info("=" * 70)
    
    # Step 1: Initialize Pixeltable
    logger.info("\n1. Initializing Pixeltable...")
    try:
        init_pixeltable()
        init_tables()
        logger.info("✓ Pixeltable initialized")
    except Exception as e:
        logger.error(f"Pixeltable init failed: {e}")
        logger.info("Skipping Pixeltable-dependent tests...")
        return
    
    # Step 2: Create sample dataset
    logger.info("\n2. Creating sample dataset...")
    dataset_id = await create_sample_dataset()
    logger.info(f"✓ Dataset created: {dataset_id}")
    
    # Step 3: Create baseline prompt
    logger.info("\n3. Creating baseline prompt...")
    prompt_id = await create_baseline_prompt()
    logger.info(f"✓ Baseline prompt: {prompt_id}")
    
    # Step 4: Configure optimizer
    logger.info("\n4. Configuring Production DSPy Optimizer...")
    config = OptimizationConfig(
        teacher_model="llama-3.3-70b-versatile",
        student_model="llama-3.1-8b-instant",
        teacher_provider="groq",
        student_provider="groq",
        optimizer_type=OptimizerType.BOOTSTRAP,
        num_trials=3,
        num_fewshot_examples=3,
        budget_usd=0.5,
        metric_name="correctness_metric"
    )
    
    optimizer = ProductionDSPyOptimizer(config)
    logger.info("✓ Optimizer configured")
    
    # Step 5: Run optimization
    logger.info("\n5. Running Bootstrap optimization...")
    logger.info("   Teacher: Groq llama-3.3-70b (generates examples)")
    logger.info("   Student: Groq llama-3.1-8b (production model)")
    
    def progress_callback(pct, msg):
        logger.info(f"   [{int(pct):3d}%] {msg}")
    
    metric_fn = create_metric_function("correctness_metric")
    
    try:
        result = optimizer.optimize_with_bootstrap(
            prompt_id=prompt_id,
            dataset_id=dataset_id,
            metric_fn=metric_fn,
            progress_callback=progress_callback,
            max_demos=3
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ OPTIMIZATION SUCCESSFUL!")
        logger.info("=" * 70)
        logger.info(f"Baseline score:    {result['baseline_score']:.3f}")
        logger.info(f"Optimized score:   {result['optimized_score']:.3f}")
        logger.info(f"Improvement:       {result['improvement_pct']:+.1f}%")
        logger.info(f"Few-shot examples: {result['num_examples']}")
        logger.info(f"Artifact saved:    {result['artifact_path']}")
        logger.info(f"Student model:     {result['model_info']['student']}")
        logger.info(f"Teacher model:     {result['model_info']['teacher']}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


async def create_sample_dataset() -> str:
    """Create sample sentiment analysis dataset."""
    import uuid
    
    dataset_id = str(uuid.uuid4())
    
    samples = [
        # Train
        {"text": "I love this product!", "sentiment": "positive", "split": "train"},
        {"text": "This is terrible quality.", "sentiment": "negative", "split": "train"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral", "split": "train"},
        {"text": "Best purchase ever!", "sentiment": "positive", "split": "train"},
        {"text": "Complete waste of money.", "sentiment": "negative", "split": "train"},
        {"text": "Decent for the price.", "sentiment": "neutral", "split": "train"},
        # Dev
        {"text": "Amazing quality!", "sentiment": "positive", "split": "dev"},
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
            'dataset_name': 'Sentiment Analysis',
            'project_id': '',
            'created_at': datetime.utcnow(),
        }])
    
    return dataset_id


async def create_baseline_prompt() -> str:
    """Create baseline prompt in Pixeltable."""
    import uuid
    
    prompt_id = str(uuid.uuid4())
    
    schemas = get_schemas()
    table = schemas.get_table('prompts')
    
    table.insert([{
        'prompt_id': prompt_id,
        'version_hash': 'baseline_v1',
        'template_str': 'Classify sentiment: {text}',
        'dspy_signature': {
            'inputs': ['text'],
            'outputs': ['sentiment']
        },
        'parent_hash': '',
        'name': 'Baseline Sentiment Classifier',
        'description': 'Simple sentiment classification',
        'tags': ['baseline'],
        'created_at': datetime.utcnow(),
        'author_id': 'test_user',
        'artifact_path': '',
        'optimizer_type': '',
        'baseline_metric': None,
        'optimized_metric': None,
    }])
    
    return prompt_id


if __name__ == "__main__":
    asyncio.run(test_production_optimizer())
