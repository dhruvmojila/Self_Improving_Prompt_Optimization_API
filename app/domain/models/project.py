"""
Project domain models.
Defines schemas for project management and organization.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ProjectBase(BaseModel):
    """Base project schema with common fields."""
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=500, description="Project description")


class ProjectCreate(ProjectBase):
    """Schema for creating a new project."""
    pass


class ProjectUpdate(BaseModel):
    """Schema for updating an existing project."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None


class Project(ProjectBase):
    """Full project schema with generated fields."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Statistics
    num_prompts: int = 0
    num_datasets: int = 0
    num_optimizations: int = 0
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "proj_abc123",
                "name": "Sentiment Analysis Optimizer",
                "description": "Optimize prompts for sentiment classification",
                "created_at": "2024-01-01T00:00:00Z",
                "num_prompts": 5,
                "num_datasets": 2,
                "num_optimizations": 3
            }
        }


class ProjectList(BaseModel):
    """Paginated list of projects."""
    projects: List[Project]
    total: int
    page: int = 1
    page_size: int = 20
