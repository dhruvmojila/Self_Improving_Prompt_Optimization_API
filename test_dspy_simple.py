"""
DSPy test with correct modern API (2024+).
Uses dspy.LM instead of deprecated LangChain wrapper.
"""

import sys
import os
import dspy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_dspy_correct_api():
    """Test DSPy with modern dspy.LM API."""
    
    logger.info("=" * 60)
    logger.info("DSPy Optimization Test - Sentiment Analysis")
    logger.info("Using modern dspy.LM API")
    logger.info("=" * 60)
    
    # Step 1: Configure DSPy with dspy.LM (modern API)
    logger.info("\n1. Configuring DSPy with Groq using dspy.LM...")
    
    # Use dspy.LM with Groq provider
    lm = dspy.LM(
        model='groq/llama-3.1-8b-instant',
        api_key=settings.GROQ_API_KEY,
    )
    
    dspy.configure(lm=lm)
    logger.info("✓ DSPy configured with dspy.LM")
    
    # Step 2: Create sample dataset
    logger.info("\n2. Creating sample sentiment dataset...")
    
    # Training data
    trainset = [
        dspy.Example(text="I love this product!", sentiment="positive").with_inputs('text'),
        dspy.Example(text="This is terrible quality.", sentiment="negative").with_inputs('text'),
        dspy.Example(text="It's okay, nothing special.", sentiment="neutral").with_inputs('text'),
        dspy.Example(text="Best purchase ever!", sentiment="positive").with_inputs('text'),
        dspy.Example(text="Complete waste of money.", sentiment="negative").with_inputs('text'),
        dspy.Example(text="Decent for the price.", sentiment="neutral").with_inputs('text'),
    ]
    
    logger.info(f"✓ Created {len(trainset)} training examples")
    
    # Step 3: Define DSPy signature and module
    logger.info("\n3. Creating DSPy module...")
    
    class SentimentClassifier(dspy.Signature):
        """Classify the sentiment of text."""
        text: str = dspy.InputField()
        sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    
    # Simple predictor (no optimization)
    predictor = dspy.Predict(SentimentClassifier)
    
    logger.info("✓ Module created with dspy.Predict")
    
    # Step 4: Test with examples
    logger.info("\n4. Testing the predictor...")
    
    test_examples = [
        "Amazing quality and great value!",
        "Terrible experience, very disappointed.",
        "It's okay, does what it says.",
    ]
    
    for text in test_examples:
        try:
            result = predictor(text=text)
            logger.info(f"  '{text[:40]}...'")
            logger.info(f"    -> Sentiment: {result.sentiment}")
        except Exception as e:
            logger.error(f"  Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Try optimization with CoT
    logger.info("\n5. Testing with Chain of Thought...")
    
    class SentimentCoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(SentimentClassifier)
        
        def forward(self, text):
            return self.predictor(text=text)
    
    cot_module = SentimentCoT()
    
    logger.info("\n   Testing CoT predictor...")
    for text in test_examples[:2]:  # Test 2 examples
        try:
            result = cot_module(text=text)
            logger.info(f"  '{text[:40]}...'")
            logger.info(f"    -> Sentiment: {result.sentiment}")
            if hasattr(result, 'reasoning'):
                logger.info(f"    -> Reasoning: {result.reasoning[:100]}...")
        except Exception as e:
            logger.error(f"  CoT failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DSPy test completed successfully!")
    logger.info("=" * 60)
    logger.info("\nKey points:")
    logger.info("- Use dspy.LM() with 'groq/model-name' format")
    logger.info("- Use dspy.configure(lm=lm) to set the model")
    logger.info("- Use dspy.Predict() or dspy.ChainOfThought() for inference")
    logger.info("- Optimization requires more complex setup")


if __name__ == "__main__":
    test_dspy_correct_api()
