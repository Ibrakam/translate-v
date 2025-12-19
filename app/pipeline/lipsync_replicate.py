"""
Cloud-based lip synchronization using Replicate API.
Runs video-retalking on cloud GPUs instead of local hardware.
"""

import os
import time
from pathlib import Path
from typing import Optional
import replicate

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class LipSyncError(Exception):
    """Raised when lip sync processing fails."""
    pass


class ReplicateLipSync:
    """
    Cloud-based lip synchronization using Replicate API.
    Offloads heavy GPU processing to cloud servers.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Replicate lip sync.

        Args:
            api_key: Replicate API key (or use REPLICATE_API_TOKEN env var)
        """
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")

        if not self.api_key:
            logger.warning(
                "No Replicate API key found. Set REPLICATE_API_TOKEN environment variable. "
                "Get your API key from https://replicate.com/account/api-tokens"
            )

        # Video Retalking model on Replicate (updated January 2024)
        # Cost: $0.00115 per second on A100 GPU
        self.model_version = "chenxwh/video-retalking:db5a650c807b007dc5f9e5abe27c53e1b62880d1f94d218d27ce7fa802711d67"

        logger.info("Replicate cloud lip sync initialized")

    def check_installation(self) -> bool:
        """
        Check if Replicate API is configured.

        Returns:
            True if API key is set
        """
        if not self.api_key:
            logger.error(
                "Replicate API key not configured. "
                "Set REPLICATE_API_TOKEN environment variable."
            )
            return False

        logger.info("✓ Replicate API is configured")
        return True

    @retry_on_exception(exceptions=(LipSyncError, Exception), max_retries=2)
    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Apply lip synchronization using cloud GPU.

        Args:
            video_path: Path to input video file
            audio_path: Path to new audio file (dubbed audio)
            output_path: Path for output lip-synced video
            **kwargs: Additional parameters (ignored for Replicate)

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

        if not self.check_installation():
            raise LipSyncError("Replicate API is not configured")

        logger.info(f"☁️  Uploading to cloud GPU for lip sync: {video_path.name}")
        logger.info(f"Using audio: {audio_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Upload files and run prediction
            with open(video_path, "rb") as video_file, open(audio_path, "rb") as audio_file:
                logger.info("Uploading files to Replicate...")

                # Run the model
                output = replicate.run(
                    self.model_version,
                    input={
                        "face": video_file,
                        "input_audio": audio_file
                    }
                )

                logger.info("Processing on cloud GPU...")

                # Download result
                if isinstance(output, str):
                    # Output is a URL to the result video
                    import requests
                    response = requests.get(output)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"✓ Lip sync complete: {output_path.name}")

                else:
                    raise LipSyncError(f"Unexpected output format from Replicate: {type(output)}")

            if not output_path.exists():
                raise LipSyncError(
                    f"Lip sync completed but output file not found: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Lip sync successful: {output_path.name} ({file_size_mb:.2f} MB)"
            )

            return output_path

        except replicate.exceptions.ReplicateError as e:
            error_msg = str(e)
            logger.error(f"Replicate API error: {error_msg}")
            raise LipSyncError(f"Cloud lip sync failed: {error_msg}")

        except Exception as e:
            logger.error(f"Unexpected error during cloud lip sync: {e}")
            raise LipSyncError(f"Lip sync failed: {e}")

    def process_with_fallback(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Apply lip sync with fallback to simple audio replacement.

        If cloud lip sync fails, falls back to just replacing audio without
        modifying lip movements.

        Args:
            video_path: Path to input video
            audio_path: Path to new audio
            output_path: Path for output video

        Returns:
            Path to output video
        """
        try:
            # Try cloud lip sync first
            return self.apply_lipsync(video_path, audio_path, output_path)

        except Exception as e:
            logger.warning(f"Cloud lip sync failed, falling back to audio replacement: {e}")

            # Fallback: Just replace audio without lip sync
            from app.pipeline.video_composer import VideoComposer
            composer = VideoComposer()
            return composer.replace_audio(video_path, audio_path, output_path)


# Alternative: Wav2Lip on Replicate (faster but lower quality)
class ReplicateWav2Lip:
    """
    Wav2Lip lip sync using Replicate (faster alternative).
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Replicate Wav2Lip."""
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")

        # Wav2Lip model on Replicate
        self.model_version = "devxpy/wav2lip:a6f2d70db374bfa8e81b9d27dc26eb748c6b5c0fe44a5acb46bcaa50ff95a851"

        logger.info("Replicate Wav2Lip initialized")

    def check_installation(self) -> bool:
        """Check API configuration."""
        return bool(self.api_key)

    @retry_on_exception(exceptions=(LipSyncError, Exception), max_retries=2)
    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Apply Wav2Lip cloud processing."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"☁️  Processing with Wav2Lip on cloud: {video_path.name}")

        try:
            with open(video_path, "rb") as video_file, open(audio_path, "rb") as audio_file:
                output = replicate.run(
                    self.model_version,
                    input={
                        "video": video_file,
                        "audio": audio_file
                    }
                )

                # Download result
                if isinstance(output, str):
                    import requests
                    response = requests.get(output)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"✓ Wav2Lip complete: {output_path.name}")
                    return output_path

                else:
                    raise LipSyncError(f"Unexpected output from Wav2Lip: {type(output)}")

        except Exception as e:
            logger.error(f"Wav2Lip cloud error: {e}")
            raise LipSyncError(f"Wav2Lip failed: {e}")


# Module-level convenience function
def apply_cloud_lipsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    method: str = "retalking"
) -> Path:
    """
    Convenience function to apply cloud lip sync.

    Args:
        video_path: Path to input video
        audio_path: Path to new audio
        output_path: Path for output video
        method: Lip sync method ('retalking' or 'wav2lip')

    Returns:
        Path to lip-synced video
    """
    if method == "retalking":
        processor = ReplicateLipSync()
    elif method == "wav2lip":
        processor = ReplicateWav2Lip()
    else:
        raise ValueError(f"Unknown lip sync method: {method}")

    return processor.apply_lipsync(video_path, audio_path, output_path)
