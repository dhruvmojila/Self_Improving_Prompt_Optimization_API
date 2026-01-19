"""
Optimization API endpoints.
Trigger and manage optimization jobs.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.domain.models import (
    OptimizationJob, OptimizationJobCreate, OptimizationProgress,
    OptimizationResult, OptimizationStatus, PromotionRequest, PromotionResponse
)
from app.api.v1.dependencies import get_db, get_current_user
from app.infrastructure.pixeltable import PixeltableClient
from app.core.events import event_bus

router = APIRouter()


# In-memory storage for MVP (will be replaced with Celery + Pixeltable)
_jobs_store: dict[str, OptimizationJob] = {}


@router.post("/start", response_model=OptimizationJob, status_code=status.HTTP_201_CREATED)
async def start_optimization(
    job_create: OptimizationJobCreate,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Start an optimization job.
    
    This triggers a long-running async task that will:
    1. Load the baseline prompt and dataset
    2. Run DSPy optimizer (Bootstrap or MIPRO)
    3. Evaluate candidates on dev set
    4. Return the best-performing prompt version
    
    Returns immediately with a job_id for polling.
    """
    import uuid
    from datetime import datetime
    
    job_id = str(uuid.uuid4())
    
    # Create progress tracker
    progress = OptimizationProgress(
        job_id=job_id,
        status=OptimizationStatus.PENDING,
        progress_pct=0.0,
    )
    
    # Create job
    job = OptimizationJob(
        id=job_id,
        prompt_id=job_create.prompt_id,
        dataset_id=job_create.dataset_id,
        config=job_create.config,
        status=OptimizationStatus.PENDING,
        progress=progress,
    )
    
    _jobs_store[job_id] = job
    
    # Update event bus
    event_bus.update_job_progress(
        job_id=job_id,
        status=OptimizationStatus.PENDING,
        message="Optimization job created, waiting to start"
    )
    
    # TODO: Trigger Celery task here
    # celery_tasks.run_optimization_job.delay(job_id, job_create.dict())
    
    return job


@router.get("/jobs/{job_id}", response_model=OptimizationJob)
async def get_optimization_job(
    job_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Get optimization job status.
    
    Poll this endpoint to track progress of the optimization.
    """
    if job_id not in _jobs_store:
        # Try event bus
        job_progress = event_bus.get_job_progress(job_id)
        if not job_progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Create minimal job from progress
        job = OptimizationJob(
            id=job_id,
            prompt_id="",
            dataset_id="",
            config=None,
            status=job_progress.status,
            progress=OptimizationProgress(**job_progress.dict()),
        )
        return job
    
    return _jobs_store[job_id]


@router.get("/jobs/{job_id}/result", response_model=OptimizationResult)
async def get_optimization_result(
    job_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Get final optimization result.
    
    Only available after job completes successfully.
    """
    job = await get_optimization_job(job_id, current_user, db)
    
    if job.status != OptimizationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is {job.status}, not completed"
        )
    
    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not available"
        )
    
    return job.result


@router.post("/jobs/{job_id}/promote", response_model=PromotionResponse)
async def promote_prompt(
    job_id: str,
    request: PromotionRequest,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Promote optimized prompt to production.
    
    Tags the new prompt version as 'production' and optionally
    deprecates the previous production version.
    """
    job = await get_optimization_job(job_id, current_user, db)
    
    if job.status != OptimizationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot promote: job is {job.status}"
        )
    
    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No result to promote"
        )
    
    # TODO: Update prompt tags in Pixeltable
    # - Add 'production' tag to new prompt
    # - Remove 'production' tag from old prompt if requested
    
    return PromotionResponse(
        success=True,
        new_prompt_id=job.result.optimized_prompt_id,
        previous_prompt_id=job.prompt_id,
        message=f"Promoted {job.result.optimized_prompt_id} to production"
    )


@router.get("/jobs", response_model=List[OptimizationJob])
async def list_optimization_jobs(
    status: Optional[OptimizationStatus] = None,
    limit: int = 20,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    List optimization jobs with optional filtering.
    """
    jobs = list(_jobs_store.values())
    
    if status:
        jobs = [j for j in jobs if j.status == status]
    
    return jobs[:limit]
