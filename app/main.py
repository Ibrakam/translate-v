"""
Main entry point for the video translation pipeline.
CLI interface for batch video processing.
"""

import sys
import argparse
from pathlib import Path
from typing import List
import json

from app.config import config
from app.utils.logger import setup_logger
from app.pipeline.extract_audio import AudioExtractor
from app.workers.worker import WorkerPool

# Setup main logger
logger = setup_logger(__name__, log_file=config.BASE_DIR / "logs" / "pipeline.log")


def find_video_files(input_dir: Path) -> List[Path]:
    """
    Find all supported video files in the input directory.

    Args:
        input_dir: Directory to search for videos

    Returns:
        List of video file paths
    """
    video_files = []

    for ext in config.SUPPORTED_VIDEO_FORMATS:
        video_files.extend(input_dir.glob(f"*{ext}"))

    # Sort by name for consistent processing order
    video_files.sort()

    logger.info(f"Found {len(video_files)} video files in {input_dir}")
    return video_files


def validate_environment() -> bool:
    """
    Validate that all required dependencies are available.

    Returns:
        True if environment is valid, False otherwise
    """
    logger.info("Validating environment...")

    # Check FFmpeg
    if not AudioExtractor.check_ffmpeg_available():
        logger.error("FFmpeg is not installed or not available in PATH")
        logger.error("Please install FFmpeg: https://ffmpeg.org/download.html")
        return False

    logger.info("✓ FFmpeg is available")

    # Validate config
    try:
        config.validate()
        logger.info("✓ Configuration is valid")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return False

    # Check input directory
    if not config.INPUT_DIR.exists():
        logger.error(f"Input directory not found: {config.INPUT_DIR}")
        return False

    logger.info("✓ Input directory exists")

    return True


def print_summary(results: List[dict]) -> None:
    """
    Print a summary of processing results.

    Args:
        results: List of processing result dictionaries
    """
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n✓ Successfully processed:")
        for result in successful:
            video_name = result.get('video_name', 'Unknown')
            processing_time = result.get('processing_time', 0)
            print(f"  - {video_name} ({processing_time:.2f}s)")

    if failed:
        print("\n✗ Failed to process:")
        for result in failed:
            video_name = result.get('video_name', 'Unknown')
            error = result.get('error', 'Unknown error')
            print(f"  - {video_name}: {error}")

    # Calculate total time
    total_time = sum(r.get('processing_time', 0) for r in results)
    print(f"\nTotal processing time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")

    print("=" * 70)


def save_results(results: List[dict], output_file: Path) -> None:
    """
    Save processing results to a JSON file.

    Args:
        results: List of processing result dictionaries
        output_file: Path to output JSON file
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """
    Main entry point for the video translation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Batch video translation pipeline - Generate multilingual subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in default input directory
  python -m app.main

  # Process videos from specific directory with 8 workers
  python -m app.main --input /path/to/videos --workers 8

  # Process only to Spanish
  python -m app.main --languages ES

  # Sequential processing (no parallelization)
  python -m app.main --workers 1

Environment variables (set in .env file):
  DEEPL_API_KEY      - DeepL API key (required)
  WHISPER_MODEL      - Whisper model name (default: large-v3)
  WHISPER_DEVICE     - Device for Whisper (cuda/cpu, default: cuda)
  MAX_WORKERS        - Number of parallel workers (default: 4)
  LOG_LEVEL          - Logging level (default: INFO)
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=config.INPUT_DIR,
        help=f'Input directory containing video files (default: {config.INPUT_DIR})'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=config.OUTPUT_DIR,
        help=f'Output directory for subtitles (default: {config.OUTPUT_DIR})'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=config.MAX_WORKERS,
        help=f'Number of parallel workers (default: {config.MAX_WORKERS})'
    )

    parser.add_argument(
        '--languages', '-l',
        nargs='+',
        default=list(config.TARGET_LANGUAGES.keys()),
        choices=list(config.TARGET_LANGUAGES.keys()),
        help='Target languages for translation (default: ES DE)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=config.WHISPER_MODEL,
        help=f'Whisper model to use (default: {config.WHISPER_MODEL})'
    )

    parser.add_argument(
        '--save-results',
        type=Path,
        default=None,
        help='Save processing results to JSON file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List videos that would be processed without actually processing them'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("VIDEO TRANSLATION PIPELINE")
    print("=" * 70)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target languages: {', '.join(args.languages)}")
    print(f"Workers: {args.workers}")
    print(f"Whisper model: {args.model}")
    print("=" * 70)

    # Update config with CLI arguments
    config.INPUT_DIR = args.input
    config.OUTPUT_DIR = args.output
    config.MAX_WORKERS = args.workers
    config.WHISPER_MODEL = args.model

    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    # Find video files
    video_files = find_video_files(config.INPUT_DIR)

    if not video_files:
        logger.warning(f"No video files found in {config.INPUT_DIR}")
        logger.info(f"Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS)}")
        sys.exit(0)

    # Dry run mode
    if args.dry_run:
        print("\nVideos to be processed:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video.name}")
        print(f"\nTotal: {len(video_files)} videos")
        sys.exit(0)

    # Start processing
    logger.info(f"Starting batch processing of {len(video_files)} videos...")
    logger.info(f"Using {args.workers} parallel workers")

    try:
        # Create worker pool
        pool = WorkerPool(num_workers=args.workers, shared_transcriber=False)

        # Start workers
        pool.start()

        # Add tasks
        num_tasks = pool.add_tasks(video_files, args.languages)
        logger.info(f"Added {num_tasks} tasks to queue")

        # Wait for completion
        results = pool.wait_for_completion()

        # Shutdown pool
        pool.shutdown(graceful=True)

        # Print summary
        print_summary(results)

        # Save results if requested
        if args.save_results:
            save_results(results, args.save_results)

        # Exit with error code if any processing failed
        failed_count = sum(1 for r in results if not r.get('success', False))
        if failed_count > 0:
            logger.warning(f"{failed_count} videos failed to process")
            sys.exit(1)

        logger.info("All videos processed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\nInterrupted! Shutting down workers...")
        if 'pool' in locals():
            pool.shutdown(graceful=False, timeout=5.0)
        sys.exit(130)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if 'pool' in locals():
            pool.shutdown(graceful=False, timeout=5.0)
        sys.exit(1)


if __name__ == "__main__":
    main()
