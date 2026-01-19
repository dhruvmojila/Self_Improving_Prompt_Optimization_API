"""
Optimization domain models.
Defines schemas for optimization jobs, configurations, and results.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class OptimizerType(str, Enum):
    """Types of DSPy optimizers."""
    BOOTSTRAP = "bootstrap"  # BootstrapFewShot - optimizes examples
    MIPRO = "mipro"  # MIPROv2 - optimizes instructions + examples
    RANDOM = "random"  # Random search baseline


class OptimizationStatus(str, Enum):
    """Optimization job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationConfig(BaseModel):
    """Configuration for an optimization job."""
    
    # Model selection
    teacher_model: str = Field(..., description="Teacher model for generating traces")
    student_model: str = Field(..., description="Student model for final inference")
    teacher_provider: str = Field(default="groq", description="Provider for teacher (groq/gemini)")
    student_provider: str = Field(default="groq", description="Provider for student")
    
    # Optimizer settings
    optimizer_type: OptimizerType = Field(default=OptimizerType.BOOTSTRAP)
    num_trials: Optional[int] = Field(default=10, ge=1, le=100, description="Number of optimization trials")
    num_fewshot_examples: int = Field(default=3, ge=0, le=10, description="Max few-shot examples")
    
    # Budget controls
    budget_usd: float = Field(default=5.0, ge=0.1, le=100.0, description="Maximum budget in USD")
    max_iterations: Optional[int] = Field(default=50, ge=1, description="Max optimization iterations")
    
    # Metric configuration
    metric_name: str = Field(default="correctness_metric", description="Evaluation metric to optimize")
    
    # Evaluation settings
    use_dev_set: bool = Field(default=True, description="Use dev set for optimization")
    use_test_set: bool = Field(default=True, description="Validate on test set before promotion")
    
    # Stopping criteria
    min_improvement_threshold: float = Field(default=0.05, ge=0.0, description="Minimum improvement to promote (5%)")
    early_stopping_patience: int = Field(default=5, ge=1, description="Trials without improvement before stopping")
    
    class Config:
        json_schema_extra = {
            "example": {
                "teacher_model": "llama-3.3-70b-versatile",
                "student_model": "llama-3.1-8b-instant",
                "teacher_provider": "groq",
                "student_provider": "groq",
                "optimizer_type": "mipro",
                "num_trials": 20,
                "budget_usd": 10.0,
                "metric_name": "correctness_metric",
                "min_improvement_threshold": 0.05
            }
        }


class OptimizationJobCreate(BaseModel):
    """Schema for creating an optimization job."""
    prompt_id: str = Field(..., description="Baseline prompt to optimize")
    dataset_id: str = Field(..., description="Dataset for optimization")
    config: OptimizationConfig = Field(..., description="Optimization configuration")
    
    # Optional metadata
    name: Optional[str] = Field(None, description="Friendly name for this optimization run")
    description: Optional[str] = None


class OptimizationProgress(BaseModel):
    """Real-time progress for an optimization job."""
    job_id: str
    status: OptimizationStatus
    
    # Progress metrics
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    current_best_metric: Optional[float] = None
    iterations_completed: int = 0
    total_iterations: Optional[int] = None
    
    # Resource usage
    tokens_used: int = 0
    cost_usd: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    estimated_completion_at: Optional[datetime] = None
    
    # Status message
    message: Optional[str] = None


class OptimizationResult(BaseModel):
    """Final result of an optimization job."""
    
    # New prompt version
    optimized_prompt_id: str = Field(..., description="ID of the optimized prompt version")
    
    # Metrics comparison
    baseline_score: float = Field(..., description="Baseline score on dev set")
    optimized_score: float = Field(..., description="Optimized score on dev set")
    improvement_pct: float = Field(..., description="Percentage improvement")
    
    # Test set validation (if enabled)
    test_set_score: Optional[float] = None
    test_set_improvement_pct: Optional[float] = None
    
    # Breakdown by metric dimension
    metric_breakdown: Optional[Dict[str, float]] = None
    
    # Examples that improved/regressed
    improved_examples: List[Dict[str, Any]] = Field(default_factory=list)
    regressed_examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Changelog
    changelog: Optional[str] = Field(None, description="Natural language explanation of changes")
    
    # Optimization metadata
    trials_run: int
    best_trial_index: int
    total_cost_usd: float


class OptimizationJob(BaseModel):
    """Full optimization job schema."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    
    # References
    prompt_id: str
    dataset_id: str
    
    # Configuration
    config: OptimizationConfig
    
    # Status
    status: OptimizationStatus = OptimizationStatus.PENDING
    progress: OptimizationProgress
    
    # Result (populated when completed)
    result: Optional[OptimizationResult] = None
    
    # Error info (if failed)
    error: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class OptimizationJobList(BaseModel):
    """Paginated list of optimization jobs."""
    jobs: List[OptimizationJob]
    total: int
    page: int = 1
    page_size: int = 20


class PromotionRequest(BaseModel):
    """Request to promote an optimized prompt to production."""
    job_id: str
    
    # Promotion settings
    tag_as_production: bool = Field(default=True, description="Tag new version as 'production'")
    deprecate_previous: bool = Field(default=False, description="Mark previous production version as deprecated")
    
    # Optional notes
    notes: Optional[str] = Field(None, description="Deployment notes")


class PromotionResponse(BaseModel):
    """Response after promoting a prompt."""
    success: bool
    new_prompt_id: str
    previous_prompt_id: Optional[str] = None
    message: str
