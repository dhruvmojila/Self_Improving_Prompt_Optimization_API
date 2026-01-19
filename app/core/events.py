"""
Event bus for job status tracking and real-time updates.
Simple in-memory event emitter with WebSocket support capability.
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import asyncio
from collections import defaultdict


class JobStatus(str, Enum):
    """Status values for optimization jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Types of events that can be emitted."""
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    DATASET_INGESTED = "dataset_ingested"
    PROMPT_CREATED = "prompt_created"
    EVALUATION_COMPLETE = "evaluation_complete"


class Event(BaseModel):
    """Event model for the event bus."""
    event_type: EventType
    job_id: Optional[str] = None
    timestamp: datetime = datetime.utcnow()
    data: Dict[str, Any] = {}
    message: Optional[str] = None


class JobProgress(BaseModel):
    """Progress information for a running job."""
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    current_metric: Optional[float] = None
    iterations_completed: int = 0
    total_iterations: Optional[int] = None
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class EventBus:
    """
    Simple in-memory event bus for job status tracking.
    
    Supports:
    - Event emission and subscription
    - Job progress tracking
    - Async event handlers
    - Future: WebSocket broadcasting
    """
    
    def __init__(self):
        # Event listeners: event_type -> list of callbacks
        self._listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Job progress tracking: job_id -> JobProgress
        self._jobs: Dict[str, JobProgress] = {}
        
        # Event history (limited to last 1000 events)
        self._event_history: List[Event] = []
        self._max_history_size = 1000
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to listen for
            callback: Async or sync function to call when event occurs
        """
        self._listeners[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe a callback from an event type."""
        if callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)
    
    async def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        # Notify all listeners
        for callback in self._listeners[event.event_type]:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
    
    def emit_sync(self, event: Event) -> None:
        """Synchronous version of emit (for non-async contexts)."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        for callback in self._listeners[event.event_type]:
            if not asyncio.iscoroutinefunction(callback):
                callback(event)
    
    def update_job_progress(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_pct: Optional[float] = None,
        current_metric: Optional[float] = None,
        iterations_completed: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ) -> JobProgress:
        """
        Update job progress and emit progress event.
        
        Args:
            job_id: Job identifier
            status: New job status
            progress_pct: Progress percentage (0-100)
            current_metric: Current best metric value
            iterations_completed: Number of iterations completed
            message: Status message
            error: Error message if failed
            
        Returns:
            Updated JobProgress
        """
        # Get or create job progress
        if job_id not in self._jobs:
            self._jobs[job_id] = JobProgress(
                job_id=job_id,
                status=JobStatus.PENDING,
                started_at=datetime.utcnow()
            )
        
        job = self._jobs[job_id]
        
        # Update fields
        if status is not None:
            job.status = status
            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.utcnow()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
        
        if progress_pct is not None:
            job.progress_pct = min(100.0, max(0.0, progress_pct))
        
        if current_metric is not None:
            job.current_metric = current_metric
        
        if iterations_completed is not None:
            job.iterations_completed = iterations_completed
        
        if message is not None:
            job.message = message
        
        if error is not None:
            job.error = error
            job.status = JobStatus.FAILED
        
        # Emit progress event
        event = Event(
            event_type=EventType.JOB_PROGRESS,
            job_id=job_id,
            data=job.dict(),
            message=message
        )
        self.emit_sync(event)
        
        return job
    
    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current progress for a job."""
        return self._jobs.get(job_id)
    
    def get_recent_events(self, limit: int = 100) -> List[Event]:
        """Get recent events from history."""
        return self._event_history[-limit:]
    
    def clear_completed_jobs(self, older_than_hours: int = 24) -> int:
        """
        Clear completed jobs older than specified hours.
        
        Args:
            older_than_hours: Remove jobs completed this many hours ago
            
        Returns:
            Number of jobs removed
        """
        cutoff = datetime.utcnow().timestamp() - (older_than_hours * 3600)
        removed = 0
        
        for job_id, job in list(self._jobs.items()):
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] 
                and job.completed_at is not None
                and job.completed_at.timestamp() < cutoff):
                del self._jobs[job_id]
                removed += 1
        
        return removed


# Global event bus instance
event_bus = EventBus()


# Convenience functions
async def emit_job_started(job_id: str, message: Optional[str] = None):
    """Emit job started event."""
    event_bus.update_job_progress(job_id, status=JobStatus.RUNNING, message=message)
    await event_bus.emit(Event(
        event_type=EventType.JOB_STARTED,
        job_id=job_id,
        message=message
    ))


async def emit_job_progress(
    job_id: str,
    progress_pct: float,
    current_metric: Optional[float] = None,
    iterations_completed: Optional[int] = None,
    message: Optional[str] = None
):
    """Emit job progress event."""
    event_bus.update_job_progress(
        job_id=job_id,
        progress_pct=progress_pct,
        current_metric=current_metric,
        iterations_completed=iterations_completed,
        message=message
    )


async def emit_job_completed(job_id: str, message: Optional[str] = None):
    """Emit job completed event."""
    event_bus.update_job_progress(job_id, status=JobStatus.COMPLETED, progress_pct=100.0, message=message)
    await event_bus.emit(Event(
        event_type=EventType.JOB_COMPLETED,
        job_id=job_id,
        message=message
    ))


async def emit_job_failed(job_id: str, error: str):
    """Emit job failed event."""
    event_bus.update_job_progress(job_id, status=JobStatus.FAILED, error=error)
    await event_bus.emit(Event(
        event_type=EventType.JOB_FAILED,
        job_id=job_id,
        message=f"Job failed: {error}",
        data={"error": error}
    ))
