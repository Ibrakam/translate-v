"""
Edge TTS (Microsoft Text-to-Speech) - Free alternative to ElevenLabs.
Uses Microsoft's Azure TTS service with no API key required.
"""

import asyncio
from pathlib import Path
from typing import List, Optional
import edge_tts

from app.pipeline.srt_utils import SubtitleSegment
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception

logger = get_logger(__name__)


class EdgeTTSError(Exception):
    """Raised when Edge TTS generation fails."""
    pass


# Voice mapping for different languages
EDGE_VOICES = {
    "EN": "en-US-AriaNeural",      # English (US) - Female
    "ES": "es-ES-ElviraNeural",    # Spanish (Spain) - Female
    "DE": "de-DE-KatjaNeural",     # German - Female
    "FR": "fr-FR-DeniseNeural",    # French - Female
    "IT": "it-IT-ElsaNeural",      # Italian - Female
    "PT": "pt-BR-FranciscaNeural", # Portuguese (Brazil) - Female
    "RU": "ru-RU-SvetlanaNeural",  # Russian - Female
    "JA": "ja-JP-NanamiNeural",    # Japanese - Female
    "KO": "ko-KR-SunHiNeural",     # Korean - Female
    "ZH": "zh-CN-XiaoxiaoNeural",  # Chinese - Female
}


class EdgeTTS:
    """
    Free text-to-speech using Microsoft Edge TTS.

    Advantages:
    - Completely free, no API key required
    - High quality neural voices
    - Supports many languages
    - Fast generation
    - No rate limits for reasonable use
    """

    def __init__(self):
        """Initialize Edge TTS."""
        logger.info("Edge TTS (Microsoft) initialized - FREE service, no API key needed")

    def get_voice_for_language(self, language_code: str) -> str:
        """
        Get the best voice for a given language.

        Args:
            language_code: Two-letter language code (e.g., "ES", "DE")

        Returns:
            Voice name for Edge TTS
        """
        voice = EDGE_VOICES.get(language_code.upper(), "en-US-AriaNeural")
        logger.info(f"Selected voice for {language_code}: {voice}")
        return voice

    async def _generate_speech_async(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> Path:
        """
        Generate speech asynchronously using Edge TTS.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice name (e.g., "es-ES-ElviraNeural")
            rate: Speech rate adjustment (e.g., "+10%" for faster, "-10%" for slower)
            pitch: Pitch adjustment (e.g., "+5Hz" for higher, "-5Hz" for lower)

        Returns:
            Path to generated audio file
        """
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(output_path))
            return output_path
        except Exception as e:
            raise EdgeTTSError(f"Failed to generate speech: {e}")

    def generate_speech(
        self,
        text: str,
        output_path: Path,
        language_code: str = "EN",
        **kwargs
    ) -> Path:
        """
        Generate speech from text (synchronous wrapper).

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            language_code: Two-letter language code
            **kwargs: Additional parameters (rate, pitch)

        Returns:
            Path to generated audio file

        Raises:
            EdgeTTSError: If speech generation fails
        """
        if not text or not text.strip():
            raise EdgeTTSError("Text cannot be empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice = self.get_voice_for_language(language_code)
        rate = kwargs.get("rate", "+0%")
        pitch = kwargs.get("pitch", "+0Hz")

        logger.info(f"Generating speech for {len(text)} characters...")

        # Run async function in sync context
        try:
            asyncio.run(
                self._generate_speech_async(text, output_path, voice, rate, pitch)
            )
        except Exception as e:
            raise EdgeTTSError(f"Speech generation failed: {e}")

        if not output_path.exists():
            raise EdgeTTSError(f"Generated audio file not found: {output_path}")

        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"✓ Speech generated: {output_path.name} ({file_size_kb:.2f} KB)")

        return output_path

    async def _generate_segment_async(
        self,
        segment: SubtitleSegment,
        temp_dir: Path,
        voice: str
    ) -> Path:
        """
        Generate audio for a single subtitle segment asynchronously.

        Args:
            segment: Subtitle segment
            temp_dir: Directory for temporary files
            voice: Voice name

        Returns:
            Path to generated audio segment
        """
        segment_path = temp_dir / f"segment_{segment.index:04d}.mp3"
        await self._generate_speech_async(segment.text, segment_path, voice)
        return segment_path

    def generate_from_subtitles(
        self,
        subtitles: List[SubtitleSegment],
        output_path: Path,
        language_code: str = "EN",
        temp_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate complete audio from subtitle segments.

        Note: This concatenates all subtitle text and generates a single audio file.
        For production with accurate timing, you'd want to generate segments separately
        and combine them with pydub or ffmpeg.

        Args:
            subtitles: List of subtitle segments
            output_path: Path for final audio file
            language_code: Two-letter language code
            temp_dir: Unused (kept for API compatibility)

        Returns:
            Path to generated audio file

        Raises:
            EdgeTTSError: If audio generation fails
        """
        if not subtitles:
            raise EdgeTTSError("No subtitles provided")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_chars = sum(len(sub.text) for sub in subtitles)
        logger.info(f"Generating full audio from {len(subtitles)} segments...")
        logger.info(f"Total characters: {total_chars}")

        # Concatenate all subtitle text with spaces
        full_text = " ".join(sub.text for sub in subtitles)

        # Generate single audio file from combined text
        return self.generate_speech(
            text=full_text,
            output_path=output_path,
            language_code=language_code
        )

    def generate_full_audio(
        self,
        srt_path: Path,
        output_path: Path,
        language: str = "EN"
    ) -> Path:
        """
        Generate complete audio track from SRT file.
        Compatible with ElevenLabsTTS API.

        Args:
            srt_path: Path to SRT subtitle file
            output_path: Path for output audio file
            language: Language code

        Returns:
            Path to generated audio file
        """
        from app.pipeline.srt_utils import SRTHandler

        # Parse SRT file
        segments = SRTHandler.parse_srt(srt_path)

        logger.info(f"Generating full audio from SRT: {srt_path.name}")

        # Generate audio from segments
        return self.generate_from_subtitles(
            subtitles=segments,
            output_path=output_path,
            language_code=language
        )


# Convenience function for easy import
def generate_dubbed_audio(
    subtitles: List[SubtitleSegment],
    output_path: Path,
    language_code: str = "EN"
) -> Path:
    """
    Convenience function to generate dubbed audio from subtitles.

    Args:
        subtitles: List of subtitle segments
        output_path: Path for output audio file
        language_code: Two-letter language code

    Returns:
        Path to generated audio file
    """
    tts = EdgeTTS()
    return tts.generate_from_subtitles(subtitles, output_path, language_code)


if __name__ == "__main__":
    # Quick test
    import tempfile

    tts = EdgeTTS()

    # Test simple speech generation
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output = Path(f.name)

    try:
        tts.generate_speech(
            "Hola, esto es una prueba de Edge TTS en español.",
            output,
            language_code="ES"
        )
        print(f"✓ Test successful: {output}")
        print(f"  File size: {output.stat().st_size / 1024:.2f} KB")
    finally:
        if output.exists():
            output.unlink()
