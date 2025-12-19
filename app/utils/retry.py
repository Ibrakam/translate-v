"""
Retry logic utilities for handling transient failures.
Implements exponential backoff and configurable retry strategies.
"""

import time
import functools
from typing import Callable, TypeVar, Any, Type, Tuple

from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_on_exception(
    max_retries: int = None,
    delay: int = None,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on specified exceptions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default from config)
        delay: Initial delay between retries in seconds (default from config)
        backoff: Multiplier for exponential backoff (default 2.0)
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_exception(max_retries=3, delay=1)
        def unstable_api_call():
            return requests.get("https://api.example.com")
    """
    max_retries = max_retries or config.MAX_RETRIES
    delay = delay or config.RETRY_DELAY

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            # This should never be reached, but for type safety
            raise last_exception  # type: ignore

        return wrapper

    return decorator


def retry_async_on_exception(
    max_retries: int = None,
    delay: int = None,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of retry decorator for async functions.

    Args:
        max_retries: Maximum number of retry attempts (default from config)
        delay: Initial delay between retries in seconds (default from config)
        backoff: Multiplier for exponential backoff (default 2.0)
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated async function with retry logic
    """
    import asyncio

    max_retries = max_retries or config.MAX_RETRIES
    delay = delay or config.RETRY_DELAY

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Async function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"Async function {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception  # type: ignore

        return wrapper

    return decorator
