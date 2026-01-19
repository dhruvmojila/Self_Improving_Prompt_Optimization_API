"""
Pixeltable table schema definitions.
Defines the core tables using namespace prefixes for organization.
"""

import pixeltable as pxt
from typing import Optional
import logging

from app.infrastructure.pixeltable.client import get_pixeltable_client

logger = logging.getLogger(__name__)


class PixeltableSchemas:
    """
    Schema definitions and table management for Pixeltable.
    
    Tables are organized using namespace prefixes:
    - prompts.registry
    - datasets.data  
    - evaluations.results
    - optimization.runs
    """
    
    def __init__(self):
        self.client = get_pixeltable_client()
    
    def create_prompts_table(self) -> pxt.Table:
        """
        Create Prompts Registry table.
        
        Path: prompts.registry
        Stores versioned prompts with DAG structure via parent_hash.
        """
        schema = {
            # Core fields
            'prompt_id': pxt.String,
            'version_hash': pxt.String,
            'template_str': pxt.String,
            
            # DSPy signature (serialized as JSON string)
            'dspy_signature': pxt.Json,
            
            # Versioning (DAG structure)
            'parent_hash': pxt.String,  # Points to parent version
            
            # Metadata
            'name': pxt.String,
            'description': pxt.String,
            'tags': pxt.Json,  # List of tags as JSON array
            'created_at': pxt.Timestamp,
            'author_id': pxt.String,
            
            # Artifact storage
            'artifact_path': pxt.String,  # Path to serialized DSPy module
            'optimizer_type': pxt.String,  # "bootstrap" or "mipro"
            
            # Performance metrics
            'baseline_metric': pxt.Float,
            'optimized_metric': pxt.Float,
        }
        
        table = self.client.get_or_create_table(
            table_path="prompts.registry",
            schema=schema
        )
        
        logger.info("Prompts table ready: prompts.registry")
        return table
    
    def create_datasets_table(self) -> pxt.Table:
        """
        Create Datasets table.
        
        Path: datasets.data
        Stores evaluation datasets with train/dev/test splits.
        """
        schema = {
            # Core fields
            'dataset_id': pxt.String,
            'row_id': pxt.String,  # Unique ID for each row
            
            # Data
            'input_data': pxt.Json,  # Input fields as JSON
            'ground_truth': pxt.Json,  # Expected output as JSON
            
            # Metadata
            'split': pxt.String,  # "train", "dev", or "test"
            'modality': pxt.String,  # "text", "image", etc.
            
            # Dataset info
            'dataset_name': pxt.String,
            'project_id': pxt.String,
            'created_at': pxt.Timestamp,
        }
        
        table = self.client.get_or_create_table(
            table_path="datasets.data",
            schema=schema
        )
        
        logger.info("Datasets table ready: datasets.data")
        return table
    
    def create_evaluations_table(self) -> pxt.Table:
        """
        Create Evaluations table.
        
        Path: evaluations.results
        Stores evaluation results for prompt × dataset combinations.
        """
        schema = {
            # References
            'eval_id': pxt.String,
            'prompt_id': pxt.String,
            'prompt_version_hash': pxt.String,
            'dataset_id': pxt.String,
            'row_id': pxt.String,
            
            # Input/output
            'input_data': pxt.Json,
            'ground_truth': pxt.Json,
            'prediction': pxt.String,  # Model output
            
            # Evaluation metrics
            'score': pxt.Float,  # Primary metric score
            'metric_name': pxt.String,
            
            # Additional metrics
            'correctness': pxt.Float,
            'semantic_similarity': pxt.Float,
            
            # Metadata
            'split': pxt.String,
            'created_at': pxt.Timestamp,
            'execution_time_ms': pxt.Float,
        }
        
        table = self.client.get_or_create_table(
            table_path="evaluations.results",
            schema=schema
        )
        
        logger.info("Evaluations table ready: evaluations.results")
        return table
    
    def create_optimization_runs_table(self) -> pxt.Table:
        """
        Create Optimization Runs table.
        
        Path: optimization.runs
        Tracks optimization job executions and their results.
        """
        schema = {
            # Core fields
            'run_id': pxt.String,
            'job_id': pxt.String,
            'trial_index': pxt.Int,
            
            # References
            'baseline_prompt_id': pxt.String,
            'candidate_prompt_id': pxt.String,
            'dataset_id': pxt.String,
            
            # Metrics
            'baseline_score': pxt.Float,
            'candidate_score': pxt.Float,
            'improvement': pxt.Float,
            
            # Configuration
            'optimizer_type': pxt.String,
            'config_json': pxt.Json,
            
            # Status
            'status': pxt.String,  # "running", "completed", "failed"
            'created_at': pxt.Timestamp,
            'completed_at': pxt.Timestamp,
        }
        
        table = self.client.get_or_create_table(
            table_path="optimization.runs",
            schema=schema
        )
        
        logger.info("Optimization runs table ready: optimization.runs")
        return table
    
    def initialize_all_tables(self) -> dict:
        """
        Initialize all tables at once.
        
        Returns:
            Dictionary mapping table names to table objects
        """
        logger.info("Initializing all Pixeltable tables...")
        
        tables = {
            'prompts': self.create_prompts_table(),
            'datasets': self.create_datasets_table(),
            'evaluations': self.create_evaluations_table(),
            'optimization_runs': self.create_optimization_runs_table(),
        }
        
        logger.info(f"Initialized {len(tables)} tables successfully")
        return tables
    
    def get_table(self, table_name: str) -> pxt.Table:
        """
        Get a table by short name.
        
        Args:
            table_name: Short name ('prompts', 'datasets', etc.)
            
        Returns:
            Pixeltable table object
        """
        table_paths = {
            'prompts': 'prompts.registry',
            'datasets': 'datasets.data',
            'evaluations': 'evaluations.results',
            'optimization_runs': 'optimization.runs',
        }
        
        if table_name not in table_paths:
            raise ValueError(f"Unknown table: {table_name}")
        
        return pxt.get_table(table_paths[table_name])
    
    def drop_all_tables(self):
        """Drop all tables (for testing/reset)."""
        logger.warning("Dropping all tables...")
        
        table_paths = [
            'optimization.runs',
            'evaluations.results',
            'datasets.data',
            'prompts.registry',
        ]
        
        for table_path in table_paths:
            try:
                self.client.drop_table(table_path, force=True)
            except Exception as e:
                logger.debug(f"Could not drop {table_path}: {e}")


# Global schema manager instance
_schemas: Optional[PixeltableSchemas] = None


def get_schemas() -> PixeltableSchemas:
    """Get the global schema manager instance."""
    global _schemas
    if _schemas is None:
        _schemas = PixeltableSchemas()
    return _schemas


def init_tables() -> dict:
    """
    Initialize all Pixeltable tables.
    Convenience function for startup.
    
    Returns:
        Dictionary mapping table names to table objects
    """
    schemas = get_schemas()
    return schemas.initialize_all_tables()
