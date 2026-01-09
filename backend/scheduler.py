"""
Background Task Scheduler Module.
Manages periodic tasks for data generation, ML inference, and insight updates.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config import SCHEDULER_CONFIG
from utils import setup_logger

logger = setup_logger("scheduler")


class DashboardScheduler:
    """
    Manages all background tasks for the dashboard.
    
    Tasks include:
    - Real-time data generation
    - ML model inference
    - Anomaly detection
    - Insight regeneration
    - Model retraining
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        self.scheduler = BackgroundScheduler(
            timezone="UTC",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 30
            }
        )
        
        # Task callbacks
        self._tasks: Dict[str, Callable] = {}
        
        # Task results cache
        self._results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()
        
        # Statistics
        self._stats: Dict[str, Any] = {
            "jobs_executed": 0,
            "jobs_failed": 0,
            "last_run": {},
        }
        
        # Add event listeners
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        
        logger.info("DashboardScheduler initialized")
    
    def _on_job_executed(self, event) -> None:
        """Handle job execution event."""
        self._stats["jobs_executed"] += 1
        self._stats["last_run"][event.job_id] = datetime.now().isoformat()
        logger.debug(f"Job executed: {event.job_id}")
    
    def _on_job_error(self, event) -> None:
        """Handle job error event."""
        self._stats["jobs_failed"] += 1
        logger.error(f"Job failed: {event.job_id}, Error: {event.exception}")
    
    def register_task(self, name: str, callback: Callable) -> None:
        """
        Register a task callback.
        
        Args:
            name: Task name
            callback: Function to call
        """
        self._tasks[name] = callback
        logger.info(f"Registered task: {name}")
    
    def _wrap_task(self, name: str) -> Callable:
        """
        Wrap a task to capture results.
        
        Args:
            name: Task name
        
        Returns:
            Wrapped callable
        """
        def wrapper():
            try:
                if name in self._tasks:
                    result = self._tasks[name]()
                    with self._results_lock:
                        self._results[name] = {
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "success": True
                        }
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
                with self._results_lock:
                    self._results[name] = {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "success": False
                    }
        return wrapper
    
    def setup_default_jobs(self) -> None:
        """Set up default scheduled jobs based on configuration."""
        config = SCHEDULER_CONFIG
        
        # Data generation job (every N seconds)
        if "generate_data" in self._tasks:
            self.scheduler.add_job(
                self._wrap_task("generate_data"),
                IntervalTrigger(seconds=config.get("data_generation_interval", 5)),
                id="generate_data",
                name="Generate Real-Time Data",
                replace_existing=True
            )
            logger.info(f"Scheduled: generate_data every {config.get('data_generation_interval', 5)}s")
        
        # Inference job (every N seconds)
        if "run_inference" in self._tasks:
            self.scheduler.add_job(
                self._wrap_task("run_inference"),
                IntervalTrigger(seconds=config.get("inference_interval", 30)),
                id="run_inference",
                name="Run ML Inference",
                replace_existing=True
            )
            logger.info(f"Scheduled: run_inference every {config.get('inference_interval', 30)}s")
        
        # Anomaly detection job (every N seconds)
        if "detect_anomalies" in self._tasks:
            self.scheduler.add_job(
                self._wrap_task("detect_anomalies"),
                IntervalTrigger(seconds=config.get("anomaly_detection_interval", 60)),
                id="detect_anomalies",
                name="Detect Anomalies",
                replace_existing=True
            )
            logger.info(f"Scheduled: detect_anomalies every {config.get('anomaly_detection_interval', 60)}s")
        
        # Insight generation job (every N seconds)
        if "update_insights" in self._tasks:
            self.scheduler.add_job(
                self._wrap_task("update_insights"),
                IntervalTrigger(seconds=config.get("insight_generation_interval", 120)),
                id="update_insights",
                name="Update Insights",
                replace_existing=True
            )
            logger.info(f"Scheduled: update_insights every {config.get('insight_generation_interval', 120)}s")
        
        # Model retraining job (daily at specified hour)
        if "retrain_models" in self._tasks:
            self.scheduler.add_job(
                self._wrap_task("retrain_models"),
                CronTrigger(hour=config.get("model_retrain_hour", 2)),
                id="retrain_models",
                name="Retrain ML Models",
                replace_existing=True
            )
            logger.info(f"Scheduled: retrain_models daily at {config.get('model_retrain_hour', 2)}:00")
    
    def add_job(
        self,
        task_name: str,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None
    ) -> None:
        """
        Add a custom job to the scheduler.
        
        Args:
            task_name: Name of registered task
            interval_seconds: Run every N seconds
            cron_expression: Cron expression for scheduling
        """
        if task_name not in self._tasks:
            logger.error(f"Task not registered: {task_name}")
            return
        
        if interval_seconds:
            trigger = IntervalTrigger(seconds=interval_seconds)
        elif cron_expression:
            # Parse cron expression
            parts = cron_expression.split()
            trigger = CronTrigger(
                minute=parts[0] if len(parts) > 0 else "*",
                hour=parts[1] if len(parts) > 1 else "*",
                day=parts[2] if len(parts) > 2 else "*",
                month=parts[3] if len(parts) > 3 else "*",
                day_of_week=parts[4] if len(parts) > 4 else "*"
            )
        else:
            logger.error("Must specify either interval_seconds or cron_expression")
            return
        
        self.scheduler.add_job(
            self._wrap_task(task_name),
            trigger,
            id=task_name,
            name=task_name,
            replace_existing=True
        )
        logger.info(f"Added job: {task_name}")
    
    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: Job identifier
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Error removing job {job_id}: {e}")
    
    def pause_job(self, job_id: str) -> None:
        """Pause a job."""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
        except Exception as e:
            logger.error(f"Error pausing job {job_id}: {e}")
    
    def resume_job(self, job_id: str) -> None:
        """Resume a paused job."""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
        except Exception as e:
            logger.error(f"Error resuming job {job_id}: {e}")
    
    def run_job_now(self, job_id: str) -> None:
        """
        Trigger a job to run immediately.
        
        Args:
            job_id: Job identifier
        """
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now())
                logger.info(f"Triggered immediate run: {job_id}")
            else:
                # Try running the task directly
                if job_id in self._tasks:
                    self._wrap_task(job_id)()
        except Exception as e:
            logger.error(f"Error running job {job_id}: {e}")
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
    
    def stop(self, wait: bool = True) -> None:
        """
        Stop the scheduler.
        
        Args:
            wait: Whether to wait for running jobs to complete
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler stopped")
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all scheduled jobs.
        
        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "pending": job.pending,
            })
        return jobs
    
    def get_job_result(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the last result of a task.
        
        Args:
            task_name: Task name
        
        Returns:
            Result dictionary or None
        """
        with self._results_lock:
            return self._results.get(task_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "running": self.scheduler.running,
            "job_count": len(self.scheduler.get_jobs()),
            "registered_tasks": list(self._tasks.keys()),
        }


class TaskRunner:
    """
    Simple task runner for manual task execution.
    
    Useful for running tasks outside the scheduler.
    """
    
    def __init__(self):
        """Initialize the task runner."""
        self._tasks: Dict[str, Callable] = {}
        self._running_tasks: Dict[str, threading.Thread] = {}
    
    def register(self, name: str, callback: Callable) -> None:
        """Register a task."""
        self._tasks[name] = callback
    
    def run_async(self, name: str, *args, **kwargs) -> bool:
        """
        Run a task asynchronously.
        
        Args:
            name: Task name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            True if started, False if already running
        """
        if name not in self._tasks:
            logger.error(f"Task not found: {name}")
            return False
        
        if name in self._running_tasks and self._running_tasks[name].is_alive():
            logger.warning(f"Task already running: {name}")
            return False
        
        def task_wrapper():
            try:
                self._tasks[name](*args, **kwargs)
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
        
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        self._running_tasks[name] = thread
        
        return True
    
    def run_sync(self, name: str, *args, **kwargs) -> Any:
        """
        Run a task synchronously.
        
        Args:
            name: Task name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Task result
        """
        if name not in self._tasks:
            raise ValueError(f"Task not found: {name}")
        
        return self._tasks[name](*args, **kwargs)
    
    def is_running(self, name: str) -> bool:
        """Check if a task is currently running."""
        if name not in self._running_tasks:
            return False
        return self._running_tasks[name].is_alive()
    
    def wait_for(self, name: str, timeout: Optional[float] = None) -> bool:
        """
        Wait for a task to complete.
        
        Args:
            name: Task name
            timeout: Maximum seconds to wait
        
        Returns:
            True if completed, False if timeout
        """
        if name not in self._running_tasks:
            return True
        
        self._running_tasks[name].join(timeout=timeout)
        return not self._running_tasks[name].is_alive()
