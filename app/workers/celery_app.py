"""
Production-Ready Celery Application.
Handles async optimization jobs with monitoring, retry logic, and error handling.

WINDOWS COMPATIBILITY:
Run with: celery -A app.workers.celery_app worker --pool=solo --loglevel=info
The 'solo' pool is required on Windows due to billiard/multiprocessing issues.
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import logging
import sys
from datetime import timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'prompt_optimizer',
    broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
    backend=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
)

# Celery Configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Results
    result_expires=3600,  # 1 hour
    result_extended=True,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # One task at a time for long-running jobs
    
    # Retry settings  
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Rate limiting
    task_default_rate_limit='10/m',  # 10 tasks per minute max
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Time limits
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=3900,  # 1 hour 5 min hard limit
    
    # Windows compatibility - use solo pool by default
    worker_pool='solo' if sys.platform == 'win32' else 'prefork',
    worker_concurrency=1 if sys.platform == 'win32' else 4,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.workers'])


# Signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Log when task starts."""
    logger.info(f"Task {task.name}[{task_id}] started")


@task_postrun.connect
def task_postrun_handler(task_id, task, *args, **kwargs):
    """Log when task completes."""
    logger.info(f"Task {task.name}[{task_id}] completed")


@task_failure.connect
def task_failure_handler(task_id, exception, *args, **kwargs):
    """Log task failures."""
    logger.error(f"Task {task_id} failed: {exception}")


# Beat schedule for periodic tasks (if needed)
celery_app.conf.beat_schedule = {
    'cleanup-old-results': {
        'task': 'app.workers.tasks.cleanup_old_results',
        'schedule': timedelta(hours=24),  # Run daily
    },
}


if __name__ == '__main__':
    celery_app.start()
