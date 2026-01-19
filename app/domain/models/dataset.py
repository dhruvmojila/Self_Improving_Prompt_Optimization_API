"""
Dataset domain models.
Defines schemas for dataset management, ingestion, and storage.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class DatasetSplit(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    ALL = "all"


class DatasetModality(str, Enum):
    """Dataset modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class DatasetRow(BaseModel):
    """Single row in a dataset."""
    input: Dict[str, Any] = Field(..., description="Input data (e.g., {'text': 'Great product!'})")
    output: Optional[Dict[str, Any]] = Field(None, description="Expected output/ground truth (e.g., {'sentiment': 'positive'})")
    split: DatasetSplit = Field(default=DatasetSplit.TRAIN, description="Train/dev/test split")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": {"text": "I love this product!"},
                "output": {"sentiment": "positive"},
                "split": "train",
                "metadata": {"source": "amazon_reviews"}
            }
        }


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    description: Optional[str] = Field(None, max_length=500)
    modality: DatasetModality = Field(default=DatasetModality.TEXT)
    
    # Schema definition
    input_fields: List[str] = Field(..., description="List of input field names (e.g., ['text'])")
    output_fields: List[str] = Field(..., description="List of output field names (e.g., ['sentiment'])")


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    rows: Optional[List[DatasetRow]] = Field(default_factory=list, description="Initial dataset rows")


class DatasetUpload(BaseModel):
    """Schema for file upload."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    modality: DatasetModality = DatasetModality.TEXT
    
    # File info
    filename: str
    file_size_bytes: int
    
    # Schema inference
    input_fields: Optional[List[str]] = None
    output_fields: Optional[List[str]] = None
    
    # Split configuration
    train_split: float = Field(default=0.6, ge=0.0, le=1.0)
    dev_split: float = Field(default=0.2, ge=0.0, le=1.0)
    test_split: float = Field(default=0.2, ge=0.0, le=1.0)
    
    @validator('test_split')
    def validate_splits(cls, v, values):
        """Ensure splits sum to 1.0."""
        train = values.get('train_split', 0.6)
        dev = values.get('dev_split', 0.2)
        total = train + dev + v
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        return v


class Dataset(DatasetBase):
    """Full dataset schema."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistics
    total_rows: int = 0
    train_rows: int = 0
    dev_rows: int = 0
    test_rows: int = 0
    
    # Storage info
    storage_path: Optional[str] = None
    file_format: Optional[str] = None  # "csv", "jsonl", etc.
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "ds_xyz789",
                "name": "Sentiment Analysis Dataset",
                "description": "Product reviews with sentiment labels",
                "modality": "text",
                "input_fields": ["text"],
                "output_fields": ["sentiment"],
                "total_rows": 1000,
                "train_rows": 600,
                "dev_rows": 200,
                "test_rows": 200
            }
        }


class DatasetStats(BaseModel):
    """Dataset statistics."""
    total_rows: int
    train_rows: int
    dev_rows: int
    test_rows: int
    input_fields: List[str]
    output_fields: List[str]
    
    # Per-field stats
    field_stats: Optional[Dict[str, Any]] = None


class DatasetRowsResponse(BaseModel):
    """Paginated dataset rows response."""
    rows: List[DatasetRow]
    total: int
    page: int = 1
    page_size: int = 100
    split: Optional[DatasetSplit] = None
