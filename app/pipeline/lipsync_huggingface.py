"""
Free lip synchronization using Hugging Face Spaces.
No API key required - completely free!
"""

import os
import time
from pathlib import Path
from typing import Optional
from gradio_client import Client, handle_file

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class LipSyncError(Exception):
    """Raised when lip sync processing fails."""
    pass


class HuggingFaceLipSync:
    """
    Free lip synchronization using Hugging Face Spaces.
    Uses Gradio Client API to connect to public Spaces.

    Advantages:
    - Completely FREE - no API key needed
    - No rate limits
    - Multiple Space options for fallback
    - High quality results
    """

    # Available Hugging Face Spaces for lip sync (in order of preference)
    # Format: (space_name, api_name, param_names)
    AVAILABLE_SPACES = [
        ("fffiloni/LatentSync", "/main", {"video": "video_path", "audio": "audio_path"}),
        ("postfilter/video-dubber", "/predict", {"video": "video", "audio": "audio"}),
        ("hysts/Wav2Lip", "/predict", {"video": "face", "audio": "audio"}),
    ]

    def __init__(self, space_index: int = 0):
        """
        Initialize Hugging Face lip sync.

        Args:
            space_index: Index of Space to use from AVAILABLE_SPACES
        """
        self.space_index = space_index
        self.space_name, self.api_name, self.param_names = self.AVAILABLE_SPACES[space_index]
        self.client = None

        logger.info(f"Hugging Face lip sync initialized (FREE service)")
        logger.info(f"Using Space: {self.space_name}")

    def check_installation(self) -> bool:
        """
        Check if Hugging Face Spaces API is available.

        Returns:
            True if API is available
        """
        try:
            # Try to connect to the Space
            self.client = Client(self.space_name)
            logger.info(f"✓ Connected to Hugging Face Space: {self.space_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Space {self.space_name}: {e}")
            return False

    @retry_on_exception(exceptions=(LipSyncError, Exception), max_retries=2)
    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Apply lip synchronization using free Hugging Face Spaces.

        Args:
            video_path: Path to input video file
            audio_path: Path to new audio file (dubbed audio)
            output_path: Path for output lip-synced video
            **kwargs: Additional parameters

        Returns:
            Path to lip-synced video file

        Raises:
            LipSyncError: If lip sync processing fails
            FileNotFoundError: If input files don't exist
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"☁️  Processing on Hugging Face (FREE GPU): {video_path.name}")
        logger.info(f"Using audio: {audio_path.name}")
        logger.info(f"Space: {self.space_name}")

        try:
            # Connect to the Space if not already connected
            if self.client is None:
                if not self.check_installation():
                    raise LipSyncError("Failed to connect to Hugging Face Space")

            logger.info("Uploading files to Hugging Face...")
            logger.info(f"API endpoint: {self.api_name}")

            # Build kwargs with correct parameter names for this Space
            video_param = self.param_names.get("video", "video")
            audio_param = self.param_names.get("audio", "audio")

            # Prepare video input - some spaces need special format
            video_file = str(video_path.resolve())
            audio_file = str(audio_path.resolve())

            # LatentSync uses dict format for video
            if "LatentSync" in self.space_name:
                video_input = {"video": handle_file(video_file), "subtitles": None}
            else:
                video_input = handle_file(video_file)

            predict_kwargs = {
                video_param: video_input,
                audio_param: handle_file(audio_file),
                "api_name": self.api_name
            }

            logger.info(f"Calling API with params: {video_param}, {audio_param}")

            try:
                result = self.client.predict(**predict_kwargs)
            except Exception as e:
                logger.warning(f"Named params failed: {e}")
                # Try positional arguments as fallback
                result = self.client.predict(
                    video_input,
                    handle_file(audio_file),
                    api_name=self.api_name
                )

            logger.info("Processing on free GPU cloud...")

            # Handle the result
            if result is None:
                raise LipSyncError("No result returned from Hugging Face Space")

            logger.info(f"Result type: {type(result)}")

            # Extract file path from various result formats
            result_file = None

            if isinstance(result, dict):
                # LatentSync returns dict like {"video": filepath, "subtitles": None}
                result_file = result.get("video") or result.get("output") or result.get("result")
            elif isinstance(result, tuple):
                result_file = result[0]
                if isinstance(result_file, dict):
                    result_file = result_file.get("video") or result_file.get("output")
            elif isinstance(result, str):
                result_file = result

            if result_file is None:
                raise LipSyncError(f"Could not extract file from result: {result}")

            logger.info(f"Result file: {result_file}")

            # Check if result is a file path
            if isinstance(result_file, str):
                import shutil
                # Download from URL or copy from local path
                if result_file.startswith('http'):
                    import requests
                    response = requests.get(result_file)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Copy local file
                    shutil.copy(result_file, output_path)

                logger.info(f"✓ Lip sync complete: {output_path.name}")
            else:
                raise LipSyncError(f"Unexpected result format: {type(result_file)}")

            if not output_path.exists():
                raise LipSyncError(
                    f"Lip sync completed but output file not found: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"✓ FREE lip sync successful: {output_path.name} ({file_size_mb:.2f} MB)"
            )

            return output_path

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Hugging Face lip sync error: {error_msg}")

            # Try fallback to next Space
            next_index = self.space_index + 1
            if next_index < len(self.AVAILABLE_SPACES):
                fallback_space = self.AVAILABLE_SPACES[next_index][0]
                logger.warning(f"Trying fallback Space: {fallback_space}")
                try:
                    self.space_index = next_index
                    self.space_name, self.api_name, self.param_names = self.AVAILABLE_SPACES[next_index]
                    self.client = None
                    return self.apply_lipsync(video_path, audio_path, output_path, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            raise LipSyncError(f"Free lip sync failed: {error_msg}")

    def process_with_fallback(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Apply lip sync with fallback to simple audio replacement.

        Args:
            video_path: Path to input video
            audio_path: Path to new audio
            output_path: Path for output video

        Returns:
            Path to output video
        """
        try:
            # Try free lip sync first
            return self.apply_lipsync(video_path, audio_path, output_path)

        except Exception as e:
            logger.warning(f"Free lip sync failed, falling back to audio replacement: {e}")

            # Fallback: Just replace audio without lip sync
            from app.pipeline.video_composer import VideoComposer
            composer = VideoComposer()
            return composer.replace_audio(video_path, audio_path, output_path)


# Module-level convenience function
def apply_free_lipsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    space_index: int = 0
) -> Path:
    """
    Convenience function to apply free lip sync from Hugging Face.

    Args:
        video_path: Path to input video
        audio_path: Path to new audio
        output_path: Path for output video
        space_index: Index of Space to use (0 = first/default)

    Returns:
        Path to lip-synced video
    """
    processor = HuggingFaceLipSync(space_index=space_index)
    return processor.apply_lipsync(video_path, audio_path, output_path)


if __name__ == "__main__":
    # Quick test
    import tempfile
    from pathlib import Path

    print("Testing Hugging Face FREE lip sync...")

    # This is just a connection test
    lipsync = HuggingFaceLipSync()
    if lipsync.check_installation():
        print("✓ Successfully connected to Hugging Face Space (FREE)")
        print(f"  Space: {lipsync.space_name}")
    else:
        print("✗ Failed to connect to Hugging Face Space")
