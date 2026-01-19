"""
Prompt domain models.
Defines schemas for prompt versions, lineage, and diffs.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid
import hashlib


class PromptSignature(BaseModel):
    """
    Prompt signature defining input and output fields.
    Similar to DSPy signature but as a Pydantic model.
    """
    inputs: List[str] = Field(..., description="Input field names (e.g., ['text', 'context'])")
    outputs: List[str] = Field(..., description="Output field names (e.g., ['sentiment', 'confidence'])")
    
    class Config:
        json_schema_extra = {
            "example": {
                "inputs": ["text"],
                "outputs": ["sentiment"]
            }
        }


class PromptBase(BaseModel):
    """Base prompt schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Prompt name/version tag")
    template: str = Field(..., min_length=1, description="Prompt template with {variable} placeholders")
    signature: PromptSignature = Field(..., description="Input/output signature")
    
    # Optional system prompt
    system_prompt: Optional[str] = Field(None, description="System message (if separate from template)")
    
    # Metadata
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list, description="Tags like ['production', 'experiment_01']")


class PromptCreate(PromptBase):
    """Schema for creating a new prompt."""
    project_id: Optional[str] = None
    parent_id: Optional[str] = Field(None, description="Parent prompt ID for versioning")


class PromptVersion(PromptBase):
    """Full prompt version schema with versioning info."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_hash: str = Field(..., description="SHA-256 hash of template + signature")
    project_id: Optional[str] = None
    parent_hash: Optional[str] = Field(None, description="Parent version hash for DAG")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author_id: Optional[str] = Field(None, description="User who created this version")
    
    # Compiled artifact info
    dspy_artifact_path: Optional[str] = Field(None, description="Path to serialized DSPy module")
    optimizer_type: Optional[str] = Field(None, description="Optimizer used (bootstrap/mipro)")
    
    # Performance metrics
    baseline_metric: Optional[float] = None
    optimized_metric: Optional[float] = None
    
    @validator('version_hash', always=True)
    def compute_version_hash(cls, v, values):
        """Compute SHA-256 hash if not provided."""
        if v:
            return v
        
        template = values.get('template', '')
        signature = values.get('signature')
        
        if signature:
            # Create deterministic hash from template + signature
            content = f"{template}:{signature.json()}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        
        return str(uuid.uuid4())[:16]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "prompt_abc123",
                "version_hash": "7a3f9c2e8b1d4f6a",
                "name": "Sentiment Classifier v2",
                "template": "Classify the sentiment of this text: {text}",
                "signature": {
                    "inputs": ["text"],
                    "outputs": ["sentiment"]
                },
                "tags": ["production"],
                "baseline_metric": 0.65,
                "optimized_metric": 0.87
            }
        }


class PromptDiff(BaseModel):
    """Semantic diff between two prompt versions."""
    parent_hash: str
    child_hash: str
    
    # Text diffs
    instruction_diff: Optional[str] = Field(None, description="Unified diff of instructions")
    template_diff: Optional[str] = Field(None, description="Unified diff of templates")
    
    # Few-shot diffs
    examples_added: List[Dict[str, Any]] = Field(default_factory=list)
    examples_removed: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance diff
    metric_delta: Optional[float] = Field(None, description="Change in metric (positive = improvement)")
    metric_delta_pct: Optional[float] = Field(None, description="Percentage change")
    
    # Natural language explanation
    changelog: Optional[str] = Field(None, description="LLM-generated explanation of changes")


class PromptLineage(BaseModel):
    """Prompt version lineage (DAG structure)."""
    prompt_id: str
    version_history: List[PromptVersion] = Field(default_factory=list, description="Ordered list of versions")
    
    # DAG structure
    parent_child_map: Dict[str, List[str]] = Field(default_factory=dict, description="version_hash -> child hashes")
    
    # Statistics
    total_versions: int = 0
    optimization_runs: int = 0


class PromptRunRequest(BaseModel):
    """Request to run a prompt on single input (playground mode)."""
    inputs: Dict[str, Any] = Field(..., description="Input values matching prompt signature")
    
    # Optional overrides
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    
    class Config:
        json_schema_extra = {
            "example": {
                "inputs": {"text": "I love this product!"},
                "temperature": 0.7
            }
        }


class PromptRunResponse(BaseModel):
    """Response from running a prompt."""
    outputs: Dict[str, Any] = Field(..., description="Output values matching prompt signature")
    
    # Execution metadata
    model: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
