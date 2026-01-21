"""
Celery Tasks for Async Optimization Jobs.
Production-ready with error handling, progress tracking, and cleanup.
"""

from celery import Task
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.workers.celery_app import celery_app
from app.domain.services import ProductionDSPyOptimizer, create_metric_function
from app.domain.models import OptimizationConfig, OptimizationStatus
from app.core.events import event_bus

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with progress callbacks."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        event_bus.update_job_progress(
            job_id=kwargs.get('job_id', task_id),
            status=OptimizationStatus.FAILED,
            message=f"Task failed: {str(exc)}"
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {task_id} succeeded")
        event_bus.update_job_progress(
            job_id=kwargs.get('job_id', task_id),
            status=OptimizationStatus.COMPLETED,
            message="Optimization completed successfully"
        )


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='app.workers.tasks.run_optimization_job',
    max_retries=3,
    default_retry_delay=60
)
def run_optimization_job(
    self,
    job_id: str,
    prompt_id: str,
    dataset_id: str,
    config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run optimization job asynchronously.
    
    Args:
        job_id: Unique job identifier
        prompt_id: Prompt ID to optimize
        dataset_id: Dataset ID for training
        config_dict: Optimization configuration as dict
        
    Returns:
        Optimization result dictionary
    """
    logger.info(f"Starting optimization job {job_id}")
    
    try:
        # Update status to running
        event_bus.update_job_progress(
            job_id=job_id,
            status=OptimizationStatus.RUNNING,
            progress_pct=0.0,
            message="Initializing optimizer..."
        )
        
        # Create config from dict
        config = OptimizationConfig(**config_dict)
        
        # Initialize optimizer
        optimizer = ProductionDSPyOptimizer(config)
        
        # Progress callback
        def progress_callback(pct: float, msg: str):
            event_bus.update_job_progress(
                job_id=job_id,
                status=OptimizationStatus.RUNNING,
                progress_pct=pct,
                message=msg
            )
            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': pct,
                    'message': msg,
                    'job_id': job_id
                }
            )
        
        # Create metric function
        metric_fn = create_metric_function(config.metric_name)
        
        # Run optimization
        result = optimizer.optimize_with_bootstrap(
            prompt_id=prompt_id,
            dataset_id=dataset_id,
            metric_fn=metric_fn,
            progress_callback=progress_callback,
            max_demos=config.num_fewshot_examples
        )
        
        # Add job metadata
        result['job_id'] = job_id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        
        # Update event bus
        event_bus.update_job_progress(
            job_id=job_id,
            status=OptimizationStatus.FAILED,
            message=f"Error: {str(e)}"
        )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e)
        else:
            logger.error(f"Job {job_id} failed after {self.max_retries} retries")
            raise


@celery_app.task(name='app.workers.tasks.cleanup_old_results')
def cleanup_old_results():
    """
    Periodic task to clean up old optimization results.
    Runs daily via Celery Beat.
    """
    logger.info("Running cleanup of old optimization results...")
    
    try:
        # Clear completed jobs older than 24 hours
        event_bus.clear_completed_jobs(older_than_hours=24)
        
        # TODO: Clean up old artifacts from filesystem
        # from pathlib import Path
        # artifacts_dir = Path("./artifacts/prompts")
        # ...
        
        logger.info("Cleanup completed successfully")
        return {"status": "success", "cleaned": True}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name='app.workers.tasks.health_check')
def health_check() -> Dict[str, str]:
    """
    Health check task for monitoring worker status.
    
    Returns:
        Status dictionary
    """
    return {
        "status": "healthy",
        "worker": "active",
        "timestamp": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    name='app.workers.tasks.batch_optimize',
    max_retries=2
)
def batch_optimize(
    self,
    job_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run multiple optimization jobs in batch.
    
    Args:
        job_configs: List of job configuration dictionaries
        
    Returns:
        Batch results
    """
    logger.info(f"Starting batch optimization with {len(job_configs)} jobs")
    
    results = []
    failed = []
    
    for i, config in enumerate(job_configs):
        try:
            # Update progress
            progress = (i / len(job_configs)) * 100
            self.update_state(
                state='PROGRESS',
                meta={'progress': progress, 'current': i, 'total': len(job_configs)}
            )
            
            # Run individual job
            result = run_optimization_job.apply_async(
                kwargs=config,
                expires=3600  # 1 hour expiry
            )
            
            results.append({
                'job_id': config.get('job_id'),
                'task_id': result.id,
                'status': 'submitted'
            })
            
        except Exception as e:
            logger.error(f"Failed to submit job {i}: {e}")
            failed.append({
                'job_id': config.get('job_id'),
                'error': str(e)
            })
    
    return {
        'total': len(job_configs),
        'submitted': len(results),
        'failed': len(failed),
        'results': results,
        'errors': failed
    }
