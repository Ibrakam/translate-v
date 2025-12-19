"""
Audio transcription module using OpenAI Whisper.
Converts audio to text with accurate timestamps for subtitle generation.
"""

from pathlib import Path
from typing import List, Optional
import warnings

# Suppress Whisper warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import whisper
from whisper.utils import get_writer

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception
from app.pipeline.srt_utils import SubtitleSegment, SRTHandler

logger = get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class WhisperTranscriber:
    """
    Handles audio transcription using OpenAI Whisper model.
    Generates accurate timestamps for subtitle creation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, large-v3)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self.model: Optional[whisper.Whisper] = None

        logger.info(f"Initializing Whisper transcriber with model: {self.model_name}")

    def load_model(self) -> None:
        """
        Load the Whisper model into memory.
        This can take some time for large models.
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            logger.info(f"Loading Whisper model '{self.model_name}' on device '{self.device}'...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise TranscriptionError(f"Model loading failed: {e}")

    @retry_on_exception(exceptions=(TranscriptionError, RuntimeError))
    def transcribe_audio(
        self,
        audio_path: Path,
        language: str = "en"
    ) -> dict:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file (WAV format recommended)
            language: Language code (default: 'en' for English)

        Returns:
            Dictionary containing transcription results with segments

        Raises:
            TranscriptionError: If transcription fails
            FileNotFoundError: If audio file doesn't exist
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio: {audio_path.name}")

        try:
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False,
                word_timestamps=True
            )

            # Log transcription stats
            num_segments = len(result.get("segments", []))
            duration = result.get("segments", [{}])[-1].get("end", 0) if num_segments > 0 else 0

            logger.info(
                f"Transcription complete: {num_segments} segments, "
                f"{duration:.2f}s duration"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path.name}: {e}")
            raise TranscriptionError(f"Transcription failed: {e}")

    def transcription_to_srt_segments(self, transcription: dict) -> List[SubtitleSegment]:
        """
        Convert Whisper transcription result to SRT subtitle segments.

        Args:
            transcription: Whisper transcription result dictionary

        Returns:
            List of SubtitleSegment objects
        """
        segments: List[SubtitleSegment] = []

        for i, segment in enumerate(transcription.get("segments", []), 1):
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            text = segment["text"].strip()

            srt_segment = SubtitleSegment(
                index=i,
                start_time=start_time,
                end_time=end_time,
                text=text
            )
            segments.append(srt_segment)

        return segments

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def transcribe_and_save_srt(
        self,
        audio_path: Path,
        output_path: Path,
        language: str = "en"
    ) -> Path:
        """
        Transcribe audio and save directly to SRT file.

        Args:
            audio_path: Path to audio file
            output_path: Path where SRT file will be saved
            language: Language code for transcription

        Returns:
            Path to the saved SRT file
        """
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path, language)

        # Convert to SRT segments
        srt_segments = self.transcription_to_srt_segments(transcription)

        # Save to file
        SRTHandler.write_srt(srt_segments, output_path)

        logger.info(f"SRT file saved: {output_path}")
        return output_path

    def unload_model(self) -> None:
        """
        Unload the model from memory to free up resources.
        Useful when processing is complete.
        """
        if self.model is not None:
            del self.model
            self.model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")


# Module-level convenience function
def transcribe_audio_to_srt(
    audio_path: Path,
    output_path: Path,
    model_name: Optional[str] = None,
    language: str = "en"
) -> Path:
    """
    Convenience function to transcribe audio and save as SRT.

    Args:
        audio_path: Path to audio file
        output_path: Path for output SRT file
        model_name: Optional Whisper model name
        language: Language code (default: 'en')

    Returns:
        Path to the saved SRT file
    """
    transcriber = WhisperTranscriber(model_name=model_name)
    return transcriber.transcribe_and_save_srt(audio_path, output_path, language)
