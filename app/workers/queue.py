"""
Task queue management for parallel video processing.
Coordinates work distribution across multiple worker processes.
"""

from pathlib import Path
from typing import List, Optional
from multiprocessing import Queue, Manager
from dataclasses import dataclass, asdict
import json

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoTask:
    """
    Represents a video processing task.
    """
    video_path: Path
    target_languages: List[str]
    task_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert task to dictionary."""
        return {
            "video_path": str(self.video_path),
            "target_languages": self.target_languages,
            "task_id": self.task_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoTask":
        """Create task from dictionary."""
        return cls(
            video_path=Path(data["video_path"]),
            target_languages=data["target_languages"],
            task_id=data.get("task_id")
        )


class TaskQueue:
    """
    Manages a queue of video processing tasks.
    Thread-safe for use with multiprocessing.
    """

    def __init__(self):
        """Initialize the task queue."""
        self.manager = Manager()
        self.task_queue: Queue = self.manager.Queue()
        self.result_queue: Queue = self.manager.Queue()
        self.task_counter = 0

        logger.info("Task queue initialized")

    def add_task(self, video_path: Path, target_languages: List[str]) -> int:
        """
        Add a video processing task to the queue.

        Args:
            video_path: Path to video file
            target_languages: List of language codes to translate to

        Returns:
            Task ID
        """
        self.task_counter += 1
        task = VideoTask(
            video_path=video_path,
            target_languages=target_languages,
            task_id=self.task_counter
        )

        # Convert to dict for serialization across processes
        self.task_queue.put(task.to_dict())

        logger.debug(f"Task {task.task_id} added: {video_path.name}")
        return task.task_id

    def add_tasks(
        self,
        video_paths: List[Path],
        target_languages: List[str]
    ) -> List[int]:
        """
        Add multiple video processing tasks to the queue.

        Args:
            video_paths: List of video file paths
            target_languages: List of language codes for all videos

        Returns:
            List of task IDs
        """
        task_ids = []

        for video_path in video_paths:
            task_id = self.add_task(video_path, target_languages)
            task_ids.append(task_id)

        logger.info(f"Added {len(task_ids)} tasks to queue")
        return task_ids

    def get_task(self, timeout: Optional[float] = None) -> Optional[VideoTask]:
        """
        Get the next task from the queue.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            VideoTask or None if queue is empty
        """
        try:
            task_dict = self.task_queue.get(timeout=timeout)
            task = VideoTask.from_dict(task_dict)
            return task
        except:
            return None

    def put_result(self, result: dict) -> None:
        """
        Put a processing result into the result queue.

        Args:
            result: Processing result dictionary
        """
        self.result_queue.put(result)

    def get_result(self, timeout: Optional[float] = None) -> Optional[dict]:
        """
        Get a processing result from the result queue.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Result dictionary or None if queue is empty
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None

    def get_all_results(self, expected_count: int, timeout: float = 1.0) -> List[dict]:
        """
        Collect all results from the result queue.

        Args:
            expected_count: Number of results to collect
            timeout: Timeout for each result retrieval

        Returns:
            List of result dictionaries
        """
        results = []

        for _ in range(expected_count):
            result = self.get_result(timeout=timeout)
            if result:
                results.append(result)
            else:
                logger.warning("Result retrieval timed out")
                break

        logger.info(f"Collected {len(results)}/{expected_count} results")
        return results

    def is_empty(self) -> bool:
        """
        Check if the task queue is empty.

        Returns:
            True if queue is empty
        """
        return self.task_queue.empty()

    def size(self) -> int:
        """
        Get approximate size of task queue.

        Returns:
            Number of tasks in queue
        """
        return self.task_queue.qsize()

    def add_poison_pills(self, num_workers: int) -> None:
        """
        Add poison pill messages to signal workers to stop.

        Args:
            num_workers: Number of worker processes
        """
        for _ in range(num_workers):
            self.task_queue.put(None)

        logger.info(f"Added {num_workers} poison pills to queue")

    def cleanup(self) -> None:
        """
        Clean up queue resources.
        """
        try:
            # Clear any remaining items
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except:
                    break

            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break

            logger.info("Task queue cleaned up")
        except Exception as e:
            logger.warning(f"Error during queue cleanup: {e}")
