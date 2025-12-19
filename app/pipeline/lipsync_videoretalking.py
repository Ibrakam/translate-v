"""
Lip synchronization module using Video Retalking.
Synchronizes lip movements with new audio track.
"""

import subprocess
from pathlib import Path
from typing import Optional
import shutil

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class LipSyncError(Exception):
    """Raised when lip sync processing fails."""
    pass


class VideoRetalkingLipSync:
    """
    Lip synchronization using Video Retalking model.
    Generates realistic lip movements matching the audio.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize Video Retalking lip sync processor.

        Args:
            model_path: Path to Video Retalking installation directory
        """
        self.model_path = model_path or config.VIDEO_RETALKING_PATH

        if not self.model_path.exists():
            logger.warning(
                f"Video Retalking not found at {self.model_path}. "
                "Please install Video Retalking first."
            )

        logger.info("Video Retalking lip sync initialized")

    def check_installation(self) -> bool:
        """
        Check if Video Retalking is properly installed.

        Returns:
            True if Video Retalking is available
        """
        # Check for inference script
        inference_script = self.model_path / "inference.py"

        if not inference_script.exists():
            logger.error(
                "Video Retalking inference.py not found. "
                "Please install from: https://github.com/OpenTalker/video-retalking"
            )
            return False

        # Check for checkpoints
        checkpoints_dir = self.model_path / "checkpoints"
        if not checkpoints_dir.exists():
            logger.error("Video Retalking checkpoints not found")
            return False

        logger.info("✓ Video Retalking is properly installed")
        return True

    @retry_on_exception(exceptions=(LipSyncError, subprocess.CalledProcessError))
    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        face_det_batch_size: int = 4,
        audio_sample_rate: int = 16000
    ) -> Path:
        """
        Apply lip synchronization to video using new audio.

        Args:
            video_path: Path to input video file
            audio_path: Path to new audio file (dubbed audio)
            output_path: Path for output lip-synced video
            face_det_batch_size: Batch size for face detection
            audio_sample_rate: Audio sample rate (default 16kHz)

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
            raise LipSyncError("Video Retalking is not properly installed")

        logger.info(f"Applying lip sync to: {video_path.name}")
        logger.info(f"Using audio: {audio_path.name}")

        # Prepare output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Video Retalking inference command
        # Navigate to Video Retalking directory and run inference
        # Use absolute paths since the subprocess runs from a different directory
        inference_cmd = [
            "python",
            "inference.py",
            "--face", str(video_path.resolve()),
            "--audio", str(audio_path.resolve()),
            "--outfile", str(output_path.resolve()),
            "--face_det_batch_size", str(face_det_batch_size)
        ]

        try:
            logger.info("Running Video Retalking inference...")
            logger.debug(f"Command: {' '.join(inference_cmd)}")

            # Run from Video Retalking directory
            result = subprocess.run(
                inference_cmd,
                cwd=str(self.model_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            if not output_path.exists():
                raise LipSyncError(
                    f"Lip sync completed but output file not found: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Lip sync successful: {output_path.name} ({file_size_mb:.2f} MB)"
            )

            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"Video Retalking error: {error_msg}")
            raise LipSyncError(f"Lip sync processing failed: {error_msg}")

        except Exception as e:
            logger.error(f"Unexpected error during lip sync: {e}")
            raise LipSyncError(f"Lip sync failed: {e}")

    def process_with_fallback(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Apply lip sync with fallback to simple audio replacement.

        If lip sync fails, falls back to just replacing audio without
        modifying lip movements.

        Args:
            video_path: Path to input video
            audio_path: Path to new audio
            output_path: Path for output video

        Returns:
            Path to output video
        """
        try:
            # Try lip sync first
            return self.apply_lipsync(video_path, audio_path, output_path)

        except Exception as e:
            logger.warning(f"Lip sync failed, falling back to audio replacement: {e}")

            # Fallback: Just replace audio without lip sync
            from app.pipeline.video_composer import VideoComposer
            composer = VideoComposer()
            return composer.replace_audio(video_path, audio_path, output_path)


class Wav2LipLipSync:
    """
    Alternative lip sync using Wav2Lip (faster but lower quality).
    Can be used as a fallback if Video Retalking is too slow.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize Wav2Lip processor.

        Args:
            model_path: Path to Wav2Lip installation directory
        """
        self.model_path = model_path or config.WAV2LIP_PATH
        logger.info("Wav2Lip lip sync initialized (alternative)")

    def check_installation(self) -> bool:
        """Check if Wav2Lip is installed."""
        inference_script = self.model_path / "inference.py"
        return inference_script.exists()

    @retry_on_exception(exceptions=(LipSyncError, subprocess.CalledProcessError))
    def apply_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        checkpoint_path: Optional[Path] = None
    ) -> Path:
        """
        Apply Wav2Lip lip synchronization.

        Args:
            video_path: Path to input video
            audio_path: Path to new audio
            output_path: Path for output video
            checkpoint_path: Path to Wav2Lip checkpoint

        Returns:
            Path to output video
        """
        if not self.check_installation():
            raise LipSyncError("Wav2Lip is not installed")

        if checkpoint_path is None:
            checkpoint_path = self.model_path / "checkpoints" / "wav2lip_gan.pth"

        logger.info(f"Applying Wav2Lip to: {video_path.name}")

        # Wav2Lip inference command
        cmd = [
            "python",
            "inference.py",
            "--checkpoint_path", str(checkpoint_path),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--pads", "0", "10", "0", "0",  # Padding for face detection
            "--nosmooth"  # Disable smoothing for faster processing
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.model_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            logger.info(f"Wav2Lip successful: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip error: {e.stderr}")
            raise LipSyncError(f"Wav2Lip failed: {e.stderr}")


# Module-level convenience function
def apply_lipsync_to_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    method: str = "videoretalking"
) -> Path:
    """
    Convenience function to apply lip sync.

    Args:
        video_path: Path to input video
        audio_path: Path to new audio
        output_path: Path for output video
        method: Lip sync method ('videoretalking' or 'wav2lip')

    Returns:
        Path to lip-synced video
    """
    if method == "videoretalking":
        processor = VideoRetalkingLipSync()
    elif method == "wav2lip":
        processor = Wav2LipLipSync()
    else:
        raise ValueError(f"Unknown lip sync method: {method}")

    return processor.apply_lipsync(video_path, audio_path, output_path)
