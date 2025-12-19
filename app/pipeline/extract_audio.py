"""
Audio extraction module using FFmpeg.
Converts video files to mono 16kHz WAV audio for transcription.
"""

import subprocess
from pathlib import Path
from typing import Optional

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""
    pass


class AudioExtractor:
    """
    Handles audio extraction from video files using FFmpeg.
    Produces standardized WAV files optimized for speech recognition.
    """

    def __init__(self):
        """Initialize the audio extractor."""
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.channels = config.AUDIO_CHANNELS
        self.threads = config.FFMPEG_THREADS

    @retry_on_exception(exceptions=(subprocess.CalledProcessError, AudioExtractionError))
    def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Extract audio from video file and convert to WAV format.

        Args:
            video_path: Path to input video file
            output_path: Optional path for output WAV file. If not provided,
                        uses temp directory from config

        Returns:
            Path to the extracted audio file

        Raises:
            AudioExtractionError: If extraction fails
            FileNotFoundError: If video file doesn't exist
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Use provided output path or generate temp path
        if output_path is None:
            video_name = video_path.stem
            output_path = config.get_temp_audio_path(video_name)

        logger.info(f"Extracting audio from {video_path.name} to {output_path}")

        # FFmpeg command for audio extraction
        # -i: input file
        # -vn: disable video recording
        # -acodec pcm_s16le: convert to 16-bit PCM WAV
        # -ar: audio sample rate
        # -ac: audio channels
        # -threads: number of threads for processing
        # -y: overwrite output file if exists
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", str(self.channels),  # Mono
            "-threads", str(self.threads),
            "-y",  # Overwrite
            str(output_path)
        ]

        try:
            # Run FFmpeg with output capture
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            if not output_path.exists():
                raise AudioExtractionError(
                    f"FFmpeg completed but output file not found: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Audio extracted successfully: {output_path.name} ({file_size_mb:.2f} MB)"
            )

            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise AudioExtractionError(f"Audio extraction failed: {error_msg}")

    def cleanup_audio(self, audio_path: Path) -> None:
        """
        Remove temporary audio file.

        Args:
            audio_path: Path to audio file to delete
        """
        try:
            if audio_path.exists():
                audio_path.unlink()
                logger.info(f"Cleaned up temporary audio file: {audio_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")

    @staticmethod
    def check_ffmpeg_available() -> bool:
        """
        Check if FFmpeg is installed and available in PATH.

        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


# Module-level function for convenience
def extract_audio_from_video(video_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convenience function to extract audio from a video file.

    Args:
        video_path: Path to input video file
        output_path: Optional path for output WAV file

    Returns:
        Path to the extracted audio file
    """
    extractor = AudioExtractor()
    return extractor.extract_audio(video_path, output_path)
