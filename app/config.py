"""
Configuration module for video translation pipeline.
Manages environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Central configuration class for the video translation pipeline.
    All settings are loaded from environment variables with sensible defaults.
    """

    # DeepL API Configuration
    DEEPL_API_KEY: str = os.getenv("DEEPL_API_KEY", "")

    # ElevenLabs API Configuration
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # TTS Provider Configuration
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "edge")  # "edge" (free) or "elevenlabs" (paid)

    # Whisper Model Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")  # cuda or cpu

    # Audio Processing Settings
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))  # mono

    # Concurrency Settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

    # Target Languages
    TARGET_LANGUAGES: dict[str, str] = {
        "ES": "spanish",
        "DE": "german"
    }

    # Supported Video Formats
    SUPPORTED_VIDEO_FORMATS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")

    # Directory Paths
    BASE_DIR: Path = Path(__file__).parent
    INPUT_DIR: Path = BASE_DIR / "storage" / "input"
    OUTPUT_DIR: Path = BASE_DIR / "storage" / "output"
    TEMP_DIR: Path = BASE_DIR / "storage" / "temp"

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Retry Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "5"))  # seconds

    # FFmpeg Configuration
    FFMPEG_THREADS: int = int(os.getenv("FFMPEG_THREADS", "0"))  # 0 = auto

    # Lip Sync Configuration
    VIDEO_RETALKING_PATH: Path = Path(os.getenv("VIDEO_RETALKING_PATH", "./models/video-retalking"))
    WAV2LIP_PATH: Path = Path(os.getenv("WAV2LIP_PATH", "./models/Wav2Lip"))
    LIPSYNC_METHOD: str = os.getenv("LIPSYNC_METHOD", "videoretalking")  # videoretalking or wav2lip

    # Output Settings
    GENERATE_SUBTITLES: bool = os.getenv("GENERATE_SUBTITLES", "true").lower() == "true"
    GENERATE_DUBBING: bool = os.getenv("GENERATE_DUBBING", "true").lower() == "true"
    APPLY_LIPSYNC: bool = os.getenv("APPLY_LIPSYNC", "true").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """
        Validate required configuration settings.
        Raises ValueError if critical settings are missing.
        """
        # Validate API keys based on enabled features
        if cls.GENERATE_SUBTITLES and not cls.DEEPL_API_KEY:
            raise ValueError(
                "DEEPL_API_KEY is required for subtitle translation. Please set it in your .env file."
            )

        if cls.GENERATE_DUBBING and cls.TTS_PROVIDER == "elevenlabs" and not cls.ELEVENLABS_API_KEY:
            raise ValueError(
                "ELEVENLABS_API_KEY is required for ElevenLabs voice dubbing. Please set it in your .env file or use TTS_PROVIDER=edge for free TTS."
            )

        # Create necessary directories
        cls.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_output_path(cls, video_name: str) -> Path:
        """
        Get the output directory path for a specific video.

        Args:
            video_name: Name of the video file (without extension)

        Returns:
            Path object for the video's output directory
        """
        output_path = cls.OUTPUT_DIR / video_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @classmethod
    def get_temp_audio_path(cls, video_name: str) -> Path:
        """
        Get the temporary audio file path for a video.

        Args:
            video_name: Name of the video file (without extension)

        Returns:
            Path object for the temporary audio file
        """
        return cls.TEMP_DIR / f"{video_name}_audio.wav"


# Singleton instance
config = Config()
