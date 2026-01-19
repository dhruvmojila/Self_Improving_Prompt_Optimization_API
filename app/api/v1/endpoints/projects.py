"""
Projects API endpoints.
CRUD operations for project management.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
import pixeltable as pxt

from app.domain.models import Project, ProjectCreate, ProjectUpdate, ProjectList
from app.api.v1.dependencies import get_db, get_current_user
from app.infrastructure.pixeltable import PixeltableClient

router = APIRouter()


# In-memory storage for MVP (replace with proper database later)
_projects_store: dict[str, Project] = {}


@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Create a new project.
    
    Projects organize prompts, datasets, and optimization runs.
    """
    new_project = Project(**project.dict())
    _projects_store[new_project.id] = new_project
    
    return new_project


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Get project by ID."""
    if project_id not in _projects_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    return _projects_store[project_id]


@router.get("/", response_model=ProjectList)
async def list_projects(
    page: int = 1,
    page_size: int = 20,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """List all projects with pagination."""
    projects = list(_projects_store.values())
    total = len(projects)
    
    # Simple pagination
    start = (page - 1) * page_size
    end = start + page_size
    paginated = projects[start:end]
    
    return ProjectList(
        projects=paginated,
        total=total,
        page=page,
        page_size=page_size
    )


@router.put("/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Update project details."""
    if project_id not in _projects_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    project = _projects_store[project_id]
    
    # Update fields
    update_data = project_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Delete a project."""
    if project_id not in _projects_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    del _projects_store[project_id]
    return None
