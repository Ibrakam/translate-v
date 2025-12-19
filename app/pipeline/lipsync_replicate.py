"""
Cloud-based lip synchronization using Replicate API.
Runs video-retalking on cloud GPUs instead of local hardware.
"""

import os
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

        # Kling lip sync model on Replicate
        # https://replicate.com/kwaivgi/kling-lip-sync
        self.model_name = "chenxwh/video-retalking:db5a650c807b007dc5f9e5abe27c53e1b62880d1f94d218d27ce7fa802711d67"

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
            logger.info("Uploading files to Replicate...")

            # Upload files first to get URLs (model expects URL strings, not file objects)
            with open(video_path, "rb") as video_file:
                video_file_obj = replicate.files.create(file=video_file)
                # urls is a dict-like object, access with ['get']
                if hasattr(video_file_obj, 'urls'):
                    video_url = video_file_obj.urls['get']
                else:
                    video_url = str(video_file_obj)
                logger.info(f"Video uploaded: {video_url}")
            
            with open(audio_path, "rb") as audio_file:
                audio_file_obj = replicate.files.create(file=audio_file)
                if hasattr(audio_file_obj, 'urls'):
                    audio_url = audio_file_obj.urls['get']
                else:
                    audio_url = str(audio_file_obj)
                logger.info(f"Audio uploaded: {audio_url}")

            logger.info("Running lip sync on Replicate...")
            output = replicate.run(
                self.model_name,
                input={
                    "face": video_url,
                    "input_audio": audio_url
                }
            )

            logger.info(f"Replicate returned output type: {type(output)}")
            logger.info(f"Replicate output value: {output}")

            # Handle different output formats from Replicate
            # New replicate library (1.0+) returns FileOutput objects
            import requests
            
            # Try to get the actual output value
            result = output
            
            # If output is a FileOutput object with read() method (replicate 1.0+)
            if hasattr(output, 'read'):
                logger.info("Output is FileOutput, reading directly...")
                with open(output_path, 'wb') as f:
                    f.write(output.read())
                logger.info(f"✓ Saved output directly to {output_path.name}")
            
            # If output is a string URL
            elif isinstance(output, str):
                logger.info(f"Output is string URL: {output}")
                response = requests.get(output)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            
            # If output is a list or iterator, get first item
            elif hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict)):
                items = list(output)
                logger.info(f"Output is iterable with {len(items)} items")
                if not items:
                    raise LipSyncError("Replicate returned empty output")
                
                first_item = items[0]
                logger.info(f"First item type: {type(first_item)}, value: {first_item}")
                
                # First item could be FileOutput or URL string
                if hasattr(first_item, 'read'):
                    with open(output_path, 'wb') as f:
                        f.write(first_item.read())
                elif isinstance(first_item, str):
                    response = requests.get(first_item)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise LipSyncError(f"Unknown item type in output: {type(first_item)}")
            
            # If output is a dict, look for URL field
            elif isinstance(output, dict):
                logger.info(f"Output is dict with keys: {output.keys()}")
                url = output.get('output') or output.get('url') or output.get('video')
                if url:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise LipSyncError(f"Could not find URL in dict output: {output}")
            
            else:
                raise LipSyncError(f"Unexpected output format: {type(output)}, value: {output}")

            logger.info(f"✓ Lip sync complete: {output_path.name}")

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

        # Kling lip sync model on Replicate
        self.model_name = "kwaivgi/kling-lip-sync"

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
            # Upload files first to get URLs
            with open(video_path, "rb") as video_file:
                video_file_obj = replicate.files.create(file=video_file)
                logger.info(f"Video file obj: {video_file_obj}")
                # urls is a dict-like object, access with ['get']
                if hasattr(video_file_obj, 'urls'):
                    video_url = video_file_obj.urls['get']
                else:
                    video_url = str(video_file_obj)
                logger.info(f"Video URL: {video_url}")
            
            with open(audio_path, "rb") as audio_file:
                audio_file_obj = replicate.files.create(file=audio_file)
                if hasattr(audio_file_obj, 'urls'):
                    audio_url = audio_file_obj.urls['get']
                else:
                    audio_url = str(audio_file_obj)
                logger.info(f"Audio URL: {audio_url}")

            output = replicate.run(
                self.model_name,
                input={
                    "video_url": video_url,
                    "audio_file": audio_url
                }
            )

            # Handle output (FileOutput or URL string)
            import requests
            
            if hasattr(output, 'read'):
                with open(output_path, 'wb') as f:
                    f.write(output.read())
            elif isinstance(output, str):
                response = requests.get(output)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise LipSyncError(f"Unexpected output from Wav2Lip: {type(output)}")

            logger.info(f"✓ Wav2Lip complete: {output_path.name}")
            return output_path

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
