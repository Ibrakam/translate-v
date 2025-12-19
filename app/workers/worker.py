"""
Worker process for parallel video processing.
Processes videos from a task queue using separate processes.
"""

from multiprocessing import Process
from typing import List, Optional
import time

from app.config import config
from app.utils.logger import get_logger
from app.pipeline.process_video import VideoProcessor
from app.pipeline.transcribe import WhisperTranscriber
from app.pipeline.translate import SubtitleTranslator
from app.workers.queue import TaskQueue, VideoTask

logger = get_logger(__name__)


class Worker:
    """
    Worker process that consumes tasks from a queue and processes videos.
    Each worker runs in a separate process.
    """

    def __init__(
        self,
        worker_id: int,
        task_queue: TaskQueue,
        shared_transcriber: bool = False
    ):
        """
        Initialize a worker.

        Args:
            worker_id: Unique identifier for this worker
            task_queue: Shared task queue
            shared_transcriber: Whether to share Whisper model across tasks
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.shared_transcriber = shared_transcriber
        self.process: Optional[Process] = None

        logger.info(f"Worker {worker_id} initialized")

    def run(self) -> None:
        """
        Main worker loop. Processes tasks until receiving poison pill.
        This method runs in a separate process.
        """
        # Initialize worker-specific logger
        worker_logger = get_logger(f"Worker-{self.worker_id}")
        worker_logger.info(f"Worker {self.worker_id} started")

        # Initialize processor components
        # Each worker gets its own instances to avoid sharing issues
        transcriber = WhisperTranscriber() if self.shared_transcriber else None
        translator = SubtitleTranslator()
        processor = VideoProcessor(
            transcriber=transcriber,
            translator=translator,
            cleanup_temp_files=True
        )

        # Load Whisper model once if sharing
        if self.shared_transcriber and transcriber:
            worker_logger.info(f"Worker {self.worker_id} loading Whisper model...")
            transcriber.load_model()

        tasks_processed = 0

        try:
            while True:
                # Get next task from queue (blocking)
                task = self.task_queue.get_task(timeout=1.0)

                # Check for poison pill (None signals shutdown)
                if task is None:
                    worker_logger.info(f"Worker {self.worker_id} received shutdown signal")
                    break

                if not task:
                    # Timeout, continue waiting
                    continue

                # Process the video
                worker_logger.info(
                    f"Worker {self.worker_id} processing task {task.task_id}: "
                    f"{task.video_path.name}"
                )

                try:
                    result = processor.process_video(
                        task.video_path,
                        task.target_languages
                    )

                    # Put result in result queue
                    result_dict = result.to_dict()
                    result_dict['worker_id'] = self.worker_id
                    result_dict['task_id'] = task.task_id

                    self.task_queue.put_result(result_dict)

                    tasks_processed += 1
                    worker_logger.info(
                        f"Worker {self.worker_id} completed task {task.task_id} "
                        f"({'success' if result.success else 'failed'})"
                    )

                except Exception as e:
                    worker_logger.error(
                        f"Worker {self.worker_id} error processing task {task.task_id}: {e}"
                    )

                    # Put error result
                    error_result = {
                        'worker_id': self.worker_id,
                        'task_id': task.task_id,
                        'video_name': task.video_path.stem,
                        'video_path': str(task.video_path),
                        'success': False,
                        'error': str(e)
                    }
                    self.task_queue.put_result(error_result)

        except KeyboardInterrupt:
            worker_logger.info(f"Worker {self.worker_id} interrupted")

        finally:
            # Cleanup
            if self.shared_transcriber and transcriber:
                transcriber.unload_model()

            worker_logger.info(
                f"Worker {self.worker_id} shutting down. "
                f"Processed {tasks_processed} tasks"
            )

    def start(self) -> None:
        """
        Start the worker process.
        """
        self.process = Process(target=self.run, name=f"Worker-{self.worker_id}")
        self.process.start()
        logger.info(f"Worker {self.worker_id} process started (PID: {self.process.pid})")

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Wait for the worker process to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self.process:
            self.process.join(timeout=timeout)

    def is_alive(self) -> bool:
        """
        Check if worker process is still running.

        Returns:
            True if worker is alive
        """
        return self.process.is_alive() if self.process else False

    def terminate(self) -> None:
        """
        Forcefully terminate the worker process.
        """
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            logger.warning(f"Worker {self.worker_id} terminated")


class WorkerPool:
    """
    Manages a pool of worker processes for parallel video processing.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        shared_transcriber: bool = False
    ):
        """
        Initialize the worker pool.

        Args:
            num_workers: Number of worker processes (default from config)
            shared_transcriber: Whether workers should share Whisper model
        """
        self.num_workers = num_workers or config.MAX_WORKERS
        self.shared_transcriber = shared_transcriber
        self.task_queue = TaskQueue()
        self.workers: List[Worker] = []

        logger.info(f"Worker pool initialized with {self.num_workers} workers")

    def start(self) -> None:
        """
        Start all worker processes.
        """
        logger.info(f"Starting {self.num_workers} workers...")

        for i in range(self.num_workers):
            worker = Worker(
                worker_id=i + 1,
                task_queue=self.task_queue,
                shared_transcriber=self.shared_transcriber
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"All {self.num_workers} workers started")

    def add_tasks(self, video_paths: List, target_languages: List[str]) -> int:
        """
        Add video processing tasks to the queue.

        Args:
            video_paths: List of video file paths
            target_languages: List of language codes to translate to

        Returns:
            Number of tasks added
        """
        task_ids = self.task_queue.add_tasks(video_paths, target_languages)
        return len(task_ids)

    def wait_for_completion(self, timeout: Optional[float] = None) -> List[dict]:
        """
        Wait for all tasks to complete and collect results.

        Args:
            timeout: Maximum time to wait for each result

        Returns:
            List of result dictionaries
        """
        # Signal workers to stop after completing tasks
        self.task_queue.add_poison_pills(self.num_workers)

        # Wait for all workers to finish
        logger.info("Waiting for all workers to complete...")
        for worker in self.workers:
            worker.join(timeout=timeout)

        # Collect all results
        results = []
        result_timeout = 1.0

        while True:
            result = self.task_queue.get_result(timeout=result_timeout)
            if result is None:
                break
            results.append(result)

        logger.info(f"Collected {len(results)} results from workers")
        return results

    def shutdown(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """
        Shut down the worker pool.

        Args:
            graceful: If True, wait for workers to finish current tasks
            timeout: Maximum time to wait for graceful shutdown
        """
        logger.info("Shutting down worker pool...")

        if graceful:
            # Add poison pills and wait
            self.task_queue.add_poison_pills(self.num_workers)

            start_time = time.time()
            for worker in self.workers:
                remaining = timeout - (time.time() - start_time)
                if remaining > 0:
                    worker.join(timeout=remaining)

        # Terminate any workers still running
        for worker in self.workers:
            if worker.is_alive():
                logger.warning(f"Terminating worker {worker.worker_id}")
                worker.terminate()

        # Cleanup queue
        self.task_queue.cleanup()

        logger.info("Worker pool shutdown complete")


def process_videos_parallel(
    video_paths: List,
    target_languages: List[str],
    num_workers: Optional[int] = None
) -> List[dict]:
    """
    Convenience function to process multiple videos in parallel.

    Args:
        video_paths: List of video file paths
        target_languages: List of language codes to translate to
        num_workers: Number of worker processes (default from config)

    Returns:
        List of processing result dictionaries
    """
    pool = WorkerPool(num_workers=num_workers)

    try:
        pool.start()
        pool.add_tasks(video_paths, target_languages)
        results = pool.wait_for_completion()
        return results

    finally:
        pool.shutdown(graceful=True)
