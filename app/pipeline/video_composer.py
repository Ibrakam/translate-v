"""
Video composition module for audio/video merging.
Handles audio replacement, mixing, and video encoding.
"""

import subprocess
from pathlib import Path
from typing import Optional

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class VideoCompositionError(Exception):
    """Raised when video composition fails."""
    pass


class VideoComposer:
    """
    Handles video composition tasks: audio replacement, mixing, encoding.
    Uses FFmpeg for all video/audio operations.
    """

    def __init__(self):
        """Initialize the video composer."""
        logger.info("Video composer initialized")

    @retry_on_exception(exceptions=(VideoCompositionError, subprocess.CalledProcessError))
    def replace_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        preserve_original_audio: bool = False,
        audio_volume: float = 1.0
    ) -> Path:
        """
        Replace video's audio track with new audio.

        Args:
            video_path: Path to input video file
            audio_path: Path to new audio file
            output_path: Path for output video
            preserve_original_audio: If True, mix with original audio
            audio_volume: Volume multiplier for new audio (0.0 to 2.0)

        Returns:
            Path to output video file

        Raises:
            VideoCompositionError: If audio replacement fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Replacing audio in: {video_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if preserve_original_audio:
            # Mix original and new audio
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-filter_complex",
                f"[0:a]volume=0.3[a1];[1:a]volume={audio_volume}[a2];[a1][a2]amix=inputs=2:duration=longest",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
                str(output_path)
            ]
        else:
            # Replace audio completely
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",  # Copy video stream without re-encoding
                "-c:a", "aac",   # Encode audio to AAC
                "-b:a", "192k",  # Audio bitrate
                "-map", "0:v:0", # Map video from first input
                "-map", "1:a:0", # Map audio from second input
                "-shortest",     # Match duration to shortest stream
                "-y",            # Overwrite output
                str(output_path)
            ]

        try:
            logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            if not output_path.exists():
                raise VideoCompositionError(
                    f"FFmpeg completed but output file not found: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Audio replaced successfully: {output_path.name} ({file_size_mb:.2f} MB)"
            )

            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise VideoCompositionError(f"Audio replacement failed: {error_msg}")

    def adjust_audio_timing(
        self,
        audio_path: Path,
        target_duration: float,
        output_path: Path,
        method: str = "stretch"
    ) -> Path:
        """
        Adjust audio timing to match target duration.

        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Path for output audio file
            method: Timing adjustment method ('stretch', 'cut', 'pad')

        Returns:
            Path to adjusted audio file
        """
        logger.info(f"Adjusting audio timing to {target_duration}s using {method}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "stretch":
            # Stretch/compress audio to match duration (changes pitch)
            # Get current duration first
            duration_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]

            try:
                result = subprocess.run(
                    duration_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

                current_duration = float(result.stdout.strip())
                tempo = current_duration / target_duration

                # Apply tempo filter
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", str(audio_path),
                    "-filter:a", f"atempo={tempo}",
                    "-y",
                    str(output_path)
                ]

                subprocess.run(ffmpeg_cmd, check=True)

            except Exception as e:
                logger.error(f"Audio timing adjustment failed: {e}")
                # Fallback: just copy the file
                import shutil
                shutil.copy(audio_path, output_path)

        elif method == "cut":
            # Cut audio to target duration
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-t", str(target_duration),
                "-c", "copy",
                "-y",
                str(output_path)
            ]
            subprocess.run(ffmpeg_cmd, check=True)

        elif method == "pad":
            # Pad audio with silence to target duration
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-af", f"apad=whole_dur={target_duration}",
                "-y",
                str(output_path)
            ]
            subprocess.run(ffmpeg_cmd, check=True)

        logger.info(f"Audio timing adjusted: {output_path}")
        return output_path

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration in seconds.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            duration = float(result.stdout.strip())
            return duration

        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0

    def extract_video_without_audio(
        self,
        video_path: Path,
        output_path: Path
    ) -> Path:
        """
        Extract video stream without audio (silent video).

        Args:
            video_path: Path to input video
            output_path: Path for output video

        Returns:
            Path to output video
        """
        logger.info(f"Extracting video stream from: {video_path.name}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-c:v", "copy",  # Copy video without re-encoding
            "-an",           # Remove audio
            "-y",
            str(output_path)
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            logger.info(f"Video stream extracted: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract video stream: {e}")
            raise VideoCompositionError(f"Video extraction failed: {e}")

    def merge_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        video_codec: str = "copy",
        audio_codec: str = "aac"
    ) -> Path:
        """
        Merge separate video and audio files.

        Args:
            video_path: Path to video file (can be silent)
            audio_path: Path to audio file
            output_path: Path for output video
            video_codec: Video codec ('copy' or codec name)
            audio_codec: Audio codec

        Returns:
            Path to output video
        """
        return self.replace_audio(
            video_path,
            audio_path,
            output_path,
            preserve_original_audio=False
        )


# Module-level convenience functions
def replace_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path
) -> Path:
    """
    Convenience function to replace video audio.

    Args:
        video_path: Path to input video
        audio_path: Path to new audio
        output_path: Path for output video

    Returns:
        Path to output video
    """
    composer = VideoComposer()
    return composer.replace_audio(video_path, audio_path, output_path)


def get_duration(video_path: Path) -> float:
    """
    Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    composer = VideoComposer()
    return composer.get_video_duration(video_path)
