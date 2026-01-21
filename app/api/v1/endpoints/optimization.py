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
    Start an async optimization job.
    
    Dispatches a Celery task that will:
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
        message="Optimization job created, dispatching to worker..."
    )
    
    # Dispatch Celery task
    try:
        from app.workers.tasks import run_optimization_job
        
        task = run_optimization_job.apply_async(
            kwargs={
                'job_id': job_id,
                'prompt_id': job_create.prompt_id,
                'dataset_id': job_create.dataset_id,
                'config_dict': job_create.config.dict()
            },
            task_id=job_id,  # Use job_id as task_id for tracking
        )
        
        # Update job with task info
        job.celery_task_id = task.id
        _jobs_store[job_id] = job
        
    except Exception as e:
        # Fallback to in-memory processing if Celery unavailable
        import logging
        logging.warning(f"Celery unavailable, job will run synchronously: {e}")
        event_bus.update_job_progress(
            job_id=job_id,
            status=OptimizationStatus.PENDING,
            message="Warning: Running without Celery (synchronous mode)"
        )
    
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
    Queries Celery/Redis for real-time task status.
    """
    # First check in-memory store for job metadata
    job = _jobs_store.get(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Query Celery for actual task status from Redis
    try:
        from celery.result import AsyncResult
        from app.workers.celery_app import celery_app
        
        task_result = AsyncResult(job_id, app=celery_app)
        
        if task_result.state == 'PENDING':
            job.status = OptimizationStatus.PENDING
            job.progress = OptimizationProgress(
                job_id=job_id,
                status=OptimizationStatus.PENDING,
                progress_pct=0.0,
                message="Waiting to start..."
            )
        elif task_result.state == 'STARTED' or task_result.state == 'PROGRESS':
            job.status = OptimizationStatus.RUNNING
            meta = task_result.info or {}
            job.progress = OptimizationProgress(
                job_id=job_id,
                status=OptimizationStatus.RUNNING,
                progress_pct=meta.get('progress', 0.0),
                message=meta.get('message', 'Running...')
            )
        elif task_result.state == 'SUCCESS':
            job.status = OptimizationStatus.COMPLETED
            result_data = task_result.result or {}
            job.progress = OptimizationProgress(
                job_id=job_id,
                status=OptimizationStatus.COMPLETED,
                progress_pct=100.0,
                message="Optimization completed!"
            )
            # Store result
            job.result = OptimizationResult(
                job_id=job_id,
                baseline_prompt_id=job.prompt_id,
                optimized_prompt_id=result_data.get('artifact_path', ''),
                baseline_score=result_data.get('baseline_score', 0),
                optimized_score=result_data.get('optimized_score', 0),
                improvement_pct=result_data.get('improvement_pct', 0),
                num_trials=result_data.get('num_trials', 0),
                best_iteration=0,
                artifact_path=result_data.get('artifact_path', ''),
                optimizer_type=result_data.get('optimizer_type', 'bootstrap'),
            )
        elif task_result.state == 'FAILURE':
            job.status = OptimizationStatus.FAILED
            job.progress = OptimizationProgress(
                job_id=job_id,
                status=OptimizationStatus.FAILED,
                progress_pct=0.0,
                message=f"Failed: {str(task_result.info)}"
            )
        
        # Update store
        _jobs_store[job_id] = job
        
    except Exception as e:
        # Fallback to in-memory status if Celery unavailable
        import logging
        logging.debug(f"Could not get Celery status: {e}")
    
    return job


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
