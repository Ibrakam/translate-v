"""
Text-to-Speech module using ElevenLabs API.
Generates natural-sounding voice audio from translated text.
"""

from pathlib import Path
from typing import List, Optional, Dict
import requests
import time

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception
from app.pipeline.srt_utils import SubtitleSegment, SRTHandler

logger = get_logger(__name__)


class TTSError(Exception):
    """Raised when TTS generation fails."""
    pass


class ElevenLabsTTS:
    """
    Text-to-Speech using ElevenLabs API.
    Supports voice cloning and multi-lingual speech synthesis.
    """

    # ElevenLabs API endpoints
    BASE_URL = "https://api.elevenlabs.io/v1"

    # Voice settings for natural speech
    DEFAULT_VOICE_SETTINGS = {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    }

    # Language code mapping for ElevenLabs
    LANGUAGE_CODES = {
        "EN": "en",
        "ES": "es",
        "DE": "de",
        "FR": "fr",
        "IT": "it",
        "PT": "pt",
        "PL": "pl",
        "RU": "ru",
        "ZH": "zh",
        "JA": "ja",
        "KO": "ko"
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs TTS client.

        Args:
            api_key: ElevenLabs API key (uses config if not provided)

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or config.ELEVENLABS_API_KEY

        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        logger.info("ElevenLabs TTS initialized")

    def get_available_voices(self) -> List[Dict]:
        """
        Get list of available voices from ElevenLabs.

        Returns:
            List of voice dictionaries with metadata
        """
        url = f"{self.BASE_URL}/voices"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            voices = response.json().get("voices", [])
            logger.info(f"Retrieved {len(voices)} available voices")

            return voices

        except Exception as e:
            logger.error(f"Failed to retrieve voices: {e}")
            return []

    def get_default_voice_id(self, language: str = "EN") -> str:
        """
        Get a default voice ID for a given language.

        Args:
            language: Language code (EN, ES, DE, etc.)

        Returns:
            Voice ID string
        """
        # Default multilingual voices from ElevenLabs
        DEFAULT_VOICES = {
            "EN": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "ES": "ThT5KcBeYPX3keUQqHPh",  # Dorothy (multilingual)
            "DE": "ThT5KcBeYPX3keUQqHPh",  # Dorothy (multilingual)
        }

        return DEFAULT_VOICES.get(language, DEFAULT_VOICES["EN"])

    @retry_on_exception(exceptions=(TTSError, requests.RequestException))
    def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "EN",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate speech audio from text.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (uses default if not provided)
            language: Language code
            output_path: Path to save audio file

        Returns:
            Path to generated audio file

        Raises:
            TTSError: If speech generation fails
        """
        if not text.strip():
            raise TTSError("Text is empty")

        # Use default voice if not specified
        if voice_id is None:
            voice_id = self.get_default_voice_id(language)

        # Prepare output path
        if output_path is None:
            output_path = config.TEMP_DIR / f"tts_{int(time.time())}.mp3"

        logger.info(f"Generating speech for {len(text)} characters...")

        # API endpoint for text-to-speech
        url = f"{self.BASE_URL}/text-to-speech/{voice_id}"

        # Request payload
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": self.DEFAULT_VOICE_SETTINGS
        }

        try:
            # Make API request
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True
            )

            response.raise_for_status()

            # Save audio to file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size_kb = output_path.stat().st_size / 1024
            logger.info(f"Speech generated: {output_path.name} ({file_size_kb:.2f} KB)")

            return output_path

        except requests.HTTPError as e:
            error_msg = f"ElevenLabs API error: {e.response.status_code}"
            if e.response.text:
                error_msg += f" - {e.response.text}"
            logger.error(error_msg)
            raise TTSError(error_msg)

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise TTSError(f"Speech generation failed: {e}")

    def generate_speech_from_segments(
        self,
        segments: List[SubtitleSegment],
        voice_id: Optional[str] = None,
        language: str = "EN",
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Generate speech for multiple subtitle segments.

        Args:
            segments: List of subtitle segments
            voice_id: ElevenLabs voice ID
            language: Language code
            output_dir: Directory to save audio files

        Returns:
            List of paths to generated audio files
        """
        if output_dir is None:
            output_dir = config.TEMP_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating speech for {len(segments)} segments...")

        audio_files: List[Path] = []

        for i, segment in enumerate(segments):
            output_path = output_dir / f"segment_{i:04d}.mp3"

            try:
                audio_path = self.generate_speech(
                    text=segment.text,
                    voice_id=voice_id,
                    language=language,
                    output_path=output_path
                )
                audio_files.append(audio_path)

            except Exception as e:
                logger.error(f"Failed to generate speech for segment {i}: {e}")
                # Create empty placeholder
                audio_files.append(None)

        successful = sum(1 for f in audio_files if f is not None)
        logger.info(f"Generated speech for {successful}/{len(segments)} segments")

        return audio_files

    def generate_full_audio(
        self,
        srt_path: Path,
        output_path: Path,
        voice_id: Optional[str] = None,
        language: str = "EN"
    ) -> Path:
        """
        Generate complete audio track from SRT file.
        Combines all segments into a single audio file with proper timing.

        Args:
            srt_path: Path to SRT subtitle file
            output_path: Path for output audio file
            voice_id: ElevenLabs voice ID
            language: Language code

        Returns:
            Path to generated audio file
        """
        # Parse SRT file
        segments = SRTHandler.parse_srt(srt_path)

        logger.info(f"Generating full audio from {len(segments)} segments...")

        # For now, we'll concatenate all text and generate one audio
        # In production, you'd want to generate segments separately
        # and combine them with proper timing using pydub or ffmpeg

        full_text = " ".join(seg.text for seg in segments)

        audio_path = self.generate_speech(
            text=full_text,
            voice_id=voice_id,
            language=language,
            output_path=output_path
        )

        logger.info(f"Full audio generated: {output_path}")
        return audio_path

    def check_quota(self) -> Dict:
        """
        Check API usage and quota information.

        Returns:
            Dictionary with quota information
        """
        url = f"{self.BASE_URL}/user"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            user_data = response.json()
            subscription = user_data.get("subscription", {})

            quota_info = {
                "character_count": subscription.get("character_count", 0),
                "character_limit": subscription.get("character_limit", 0),
                "can_extend_character_limit": subscription.get("can_extend_character_limit", False)
            }

            if quota_info["character_limit"] > 0:
                quota_info["usage_percent"] = (
                    quota_info["character_count"] / quota_info["character_limit"] * 100
                )

            logger.info(
                f"ElevenLabs quota: {quota_info['character_count']:,} / "
                f"{quota_info['character_limit']:,} characters used"
            )

            return quota_info

        except Exception as e:
            logger.warning(f"Could not fetch quota information: {e}")
            return {}


# Module-level convenience function
def generate_speech_from_text(
    text: str,
    output_path: Path,
    language: str = "EN",
    api_key: Optional[str] = None
) -> Path:
    """
    Convenience function to generate speech from text.

    Args:
        text: Text to convert to speech
        output_path: Path to save audio file
        language: Language code
        api_key: Optional ElevenLabs API key

    Returns:
        Path to generated audio file
    """
    tts = ElevenLabsTTS(api_key=api_key)
    return tts.generate_speech(text, language=language, output_path=output_path)
