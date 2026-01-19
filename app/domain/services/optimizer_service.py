"""
Optimizer Service - DSPy Integration.
Handles prompt optimization using DSPy's Bootstrap and MIPRO optimizers.
"""

import dspy
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os

from app.core.config import settings
from app.infrastructure.pixeltable import get_schemas
from app.domain.models import OptimizationConfig, OptimizerType

logger = logging.getLogger(__name__)


class DSPyOptimizer:
    """
    DSPy optimization service.
    
    Provides two optimization strategies:
    1. Bootstrap - Optimizes few-shot examples
    2. MIPRO - Optimizes instructions + examples using Bayesian search
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._setup_dspy()
    
    def _setup_dspy(self):
        """Configure DSPy with native DSPy LM classes (NOT LangChain)."""
        
        # CRITICAL: DSPy does NOT accept LangChain LLMs
        # Must use dspy.BaseLM implementations: dspy.Groq, dspy.OpenAI, etc.
        
        # Get teacher LM (DSPy native)
        self.teacher_lm = self._create_dspy_lm(
            model_name=self.config.teacher_model,
            provider=self.config.teacher_provider
        )
        
        # Get student LM (DSPy native)
        self.student_lm = self._create_dspy_lm(
            model_name=self.config.student_model,
            provider=self.config.student_provider
        )
        
        # Configure DSPy with student model as default
        dspy.settings.configure(lm=self.student_lm)
        
        logger.info(f"DSPy configured with teacher={self.config.teacher_model}, student={self.config.student_model}")
    
    def _create_dspy_lm(self, model_name: str, provider: str) -> dspy.BaseLM:
        """
        Create a DSPy-native LM instance.
        
        Args:
            model_name: Model name (e.g., "llama-3.1-8b-instant")
            provider: Provider name ("groq", "openai", "gemini", etc.)
            
        Returns:
            DSPy BaseLM instance
        """
        provider = provider.lower()
        
        if provider == "groq":
            # Use DSPy's Groq adapter
            api_key = os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
            return dspy.LM(
                model=model_name,
                api_key=api_key,
                max_tokens=1024,
            )
        
        elif provider == "openai":
            # Use DSPy's OpenAI adapter
            api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
            return dspy.OpenAI(
                model=model_name,
                api_key=api_key,
                max_tokens=1024,
            )
        
        elif provider == "google" or provider == "gemini":
            # Use DSPy's Google adapter
            api_key = os.environ.get("GOOGLE_API_KEY") or settings.GOOGLE_API_KEY
            return dspy.Google(
                model=model_name,
                api_key=api_key,
                max_output_tokens=1024,
            )
        
        elif provider == "anthropic" or provider == "claude":
            # Use DSPy's Anthropic adapter
            api_key = os.environ.get("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
            return dspy.Claude(
                model=model_name,
                api_key=api_key,
                max_tokens=1024,
            )
        
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Supported: groq, openai, google/gemini, anthropic/claude"
            )
    
    def create_dynamic_signature(self, input_fields: List[str], output_fields: List[str]) -> type:
        """
        Create a DSPy signature dynamically from field lists.
        
        Args:
            input_fields: List of input field names
            output_fields: List of output field names
            
        Returns:
            DSPy Signature class
        """
        # Build signature string
        # Format: "input1, input2 -> output1, output2"
        inputs_str = ", ".join(input_fields)
        outputs_str = ", ".join(output_fields)
        signature_str = f"{inputs_str} -> {outputs_str}"
        
        # Create signature class
        signature = dspy.Signature(signature_str)
        
        logger.info(f"Created DSPy signature: {signature_str}")
        return signature
    
    def load_dataset_from_pixeltable(
        self,
        dataset_id: str,
        split: str = "train"
    ) -> List[dspy.Example]:
        """
        Load dataset from Pixeltable and convert to DSPy Examples.
        
        Args:
            dataset_id: Dataset ID
            split: Data split (train/dev/test)
            
        Returns:
            List of DSPy Example objects
        """
        schemas = get_schemas()
        table = schemas.get_table('datasets')
        
        # Query for dataset rows
        results = table.where(
            (table.dataset_id == dataset_id) & (table.split == split)
        ).collect()
        
        if not results:
            raise ValueError(f"No data found for dataset {dataset_id} with split {split}")
        
        # Convert to DSPy Examples
        examples = []
        for row in results:
            # Merge input and output data
            example_data = {**row['input_data'], **row['ground_truth']}
            
            # Create DSPy Example
            example = dspy.Example(**example_data).with_inputs(*row['input_data'].keys())
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from {split} split")
        return examples
    
    def optimize_bootstrap(
        self,
        prompt_id: str,
        dataset_id: str,
        metric_fn: callable,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Optimize using Bootstrap (few-shot example selection).
        
        Uses teacher model to generate reasoning traces, then selects
        best k examples for the student model.
        
        Args:
            prompt_id: Baseline prompt ID
            dataset_id: Dataset ID
            metric_fn: Evaluation metric function
            progress_callback: Optional callback(progress_pct, message)
            
        Returns:
            Optimization result dictionary
        """
        logger.info("Starting Bootstrap optimization...")
        
        if progress_callback:
            progress_callback(10, "Loading dataset...")
        
        # Load data
        train_data = self.load_dataset_from_pixeltable(dataset_id, split="train")
        dev_data = self.load_dataset_from_pixeltable(dataset_id, split="dev")
        
        # Get prompt signature
        schemas = get_schemas()
        prompts_table = schemas.get_table('prompts')
        prompt_row = prompts_table.where(prompts_table.prompt_id == prompt_id).collect()[0]
        
        signature = self.create_dynamic_signature(
            input_fields=prompt_row['dspy_signature']['inputs'],
            output_fields=prompt_row['dspy_signature']['outputs']
        )
        
        if progress_callback:
            progress_callback(20, "Creating DSPy module...")
        
        # Create DSPy module
        class GenerateAnswer(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.predictor = dspy.ChainOfThought(signature)
            
            def forward(self, **kwargs):
                return self.predictor(**kwargs)
        
        module = GenerateAnswer(signature)
        
        if progress_callback:
            progress_callback(30, "Running Bootstrap optimizer with teacher model...")
        
        # Create optimizer with teacher model for generating traces
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric_fn,
            max_bootstrapped_demos=self.config.num_fewshot_examples,
            max_labeled_demos=self.config.num_fewshot_examples,
            num_candidate_programs=self.config.num_trials or 10,
            teacher_settings=dict(lm=self.teacher_lm),
        )
        
        if progress_callback:
            progress_callback(50, f"Optimizing with {len(train_data)} train examples...")
        
        # Run optimization
        try:
            optimized_module = optimizer.compile(
                module,
                trainset=train_data[:min(len(train_data), 100)],  # Limit for speed
                valset=dev_data[:min(len(dev_data), 50)]
            )
        except Exception as e:
            logger.error(f"Bootstrap optimization failed: {e}")
            raise
        
        if progress_callback:
            progress_callback(80, "Evaluating optimized module...")
        
        # Evaluate on dev set
        baseline_score = self._evaluate_module(module, dev_data, metric_fn)
        optimized_score = self._evaluate_module(optimized_module, dev_data, metric_fn)
        
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        
        if progress_callback:
            progress_callback(90, "Saving optimized prompt...")
        
        # Save optimized module
        artifact_path = f"./artifacts/prompts/optimized_{prompt_id}_{datetime.utcnow().timestamp()}.json"
        optimized_module.save(artifact_path)
        
        result = {
            'optimizer_type': 'bootstrap',
            'baseline_score': baseline_score,
            'optimized_score': optimized_score,
            'improvement_pct': improvement,
            'artifact_path': artifact_path,
            'num_examples': len(optimized_module.predictor.demos) if hasattr(optimized_module.predictor, 'demos') else 0,
            'trials_run': self.config.num_trials or 10,
        }
        
        if progress_callback:
            progress_callback(100, "Optimization complete!")
        
        logger.info(f"Bootstrap complete: {baseline_score:.3f} -> {optimized_score:.3f} ({improvement:+.1f}%)")
        return result
    
    def optimize_mipro(
        self,
        prompt_id: str,
        dataset_id: str,
        metric_fn: callable,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Optimize using MIPRO (instruction + example optimization).
        
        Uses Bayesian optimization to search instruction space and
        few-shot example combinations.
        
        Args:
            prompt_id: Baseline prompt ID
            dataset_id: Dataset ID
            metric_fn: Evaluation metric function
            progress_callback: Optional callback(progress_pct, message)
            
        Returns:
            Optimization result dictionary
        """
        logger.info("Starting MIPRO optimization...")
        
        if progress_callback:
            progress_callback(10, "Loading dataset...")
        
        # Load data
        train_data = self.load_dataset_from_pixeltable(dataset_id, split="train")
        dev_data = self.load_dataset_from_pixeltable(dataset_id, split="dev")
        
        # Ensure we have enough data
        if len(dev_data) < 3:
            raise ValueError(f"Development set too small ({len(dev_data)} examples). Need at least 3 examples.")
        
        # Get prompt signature
        schemas = get_schemas()
        prompts_table = schemas.get_table('prompts')
        prompt_row = prompts_table.where(prompts_table.prompt_id == prompt_id).collect()[0]
        
        signature = self.create_dynamic_signature(
            input_fields=prompt_row['dspy_signature']['inputs'],
            output_fields=prompt_row['dspy_signature']['outputs']
        )
        
        if progress_callback:
            progress_callback(20, "Creating DSPy module...")
        
        # Create DSPy module
        class GenerateAnswer(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.predictor = dspy.ChainOfThought(signature)
            
            def forward(self, **kwargs):
                return self.predictor(**kwargs)
        
        module = GenerateAnswer(signature)
        
        if progress_callback:
            progress_callback(30, "Initializing MIPRO with Bayesian optimization...")
        
        # MIPRO optimizer with Bayesian search
        try:
            from dspy.teleprompt import MIPROv2
            
            # Calculate appropriate minibatch size
            val_size = len(dev_data)
            minibatch_size = min(6, val_size)  # Use 6 or val_size, whichever is smaller
            
            logger.info(f"Using minibatch_size={minibatch_size} for valset of size {val_size}")
            
            # CRITICAL: minibatch_size is NOT a constructor argument for MIPROv2
            # It must be passed to compile() method
            teleprompter = MIPROv2(
                metric=metric_fn,
                auto="medium",  # Use auto mode for sensible defaults
                verbose=True,
            )
            
            if progress_callback:
                progress_callback(50, f"Running MIPRO optimization...")
            
            # Prepare datasets with size limits
            train_subset = train_data[:min(len(train_data), 100)]
            val_subset = dev_data  # Use all dev data
            
            logger.info(f"Training on {len(train_subset)} examples, validating on {len(val_subset)} examples")
            
            # CRITICAL: Pass minibatch parameters to compile() method, NOT constructor
            optimized_module = teleprompter.compile(
                module,
                trainset=train_subset,
                valset=val_subset,
                minibatch=True,  # Enable minibatching during BO trials
                minibatch_size=minibatch_size,  # Correct place for this parameter
                minibatch_full_eval_steps=5,  # Optional: full eval every 5 steps
            )
            
        except ImportError:
            logger.warning("MIPROv2 not available, falling back to Bootstrap")
            return self.optimize_bootstrap(prompt_id, dataset_id, metric_fn, progress_callback)
        except Exception as e:
            logger.error(f"MIPRO optimization failed: {e}")
            raise
        
        if progress_callback:
            progress_callback(80, "Evaluating optimized module...")
        
        # Evaluate on dev set
        baseline_score = self._evaluate_module(module, dev_data, metric_fn)
        optimized_score = self._evaluate_module(optimized_module, dev_data, metric_fn)
        
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        
        if progress_callback:
            progress_callback(90, "Saving optimized prompt...")
        
        # Save optimized module
        artifact_path = f"./artifacts/prompts/optimized_{prompt_id}_{datetime.utcnow().timestamp()}.json"
        optimized_module.save(artifact_path)
        
        result = {
            'optimizer_type': 'mipro',
            'baseline_score': baseline_score,
            'optimized_score': optimized_score,
            'improvement_pct': improvement,
            'artifact_path': artifact_path,
            'num_examples': len(optimized_module.predictor.demos) if hasattr(optimized_module.predictor, 'demos') else 0,
            'trials_run': "auto",  # MIPRO auto mode determines this
        }
        
        if progress_callback:
            progress_callback(100, "MIPRO optimization complete!")
        
        logger.info(f"MIPRO complete: {baseline_score:.3f} -> {optimized_score:.3f} ({improvement:+.1f}%)")
        return result
    
    def _evaluate_module(
        self,
        module: dspy.Module,
        dataset: List[dspy.Example],
        metric_fn: callable
    ) -> float:
        """
        Evaluate a DSPy module on a dataset.
        
        Args:
            module: DSPy module to evaluate
            dataset: List of examples
            metric_fn: Metric function
            
        Returns:
            Average metric score
        """
        scores = []
        
        for example in dataset:
            try:
                # Get prediction
                prediction = module(**example.inputs())
                
                # Calculate metric
                score = metric_fn(example, prediction)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Evaluation failed for example: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0


def create_metric_function(metric_name: str) -> callable:
    """
    Create a metric function for DSPy optimization.
    
    Args:
        metric_name: Name of the metric to use
        
    Returns:
        Metric function compatible with DSPy
    """
    from app.infrastructure.pixeltable.udfs import (
        exact_match, string_similarity, correctness_metric
    )
    
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        """DSPy-compatible metric function."""
        # Get the first output field name
        output_field = list(example.labels().keys())[0]
        
        # Get ground truth and prediction
        ground_truth = str(example.labels()[output_field])
        pred_value = str(getattr(prediction, output_field, ""))
        
        # Calculate metric based on name
        if metric_name == "exact_match":
            return float(exact_match(pred_value, ground_truth))
        elif metric_name == "string_similarity":
            return float(string_similarity(pred_value, ground_truth))
        elif metric_name == "correctness_metric":
            return float(correctness_metric(pred_value, ground_truth))
        else:
            # Default to correctness
            return float(correctness_metric(pred_value, ground_truth))
    
    return metric