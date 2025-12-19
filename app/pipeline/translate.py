"""
Subtitle translation module using DeepL API.
Translates SRT segments while preserving timestamps.
"""

from pathlib import Path
from typing import List, Optional
import deepl

from app.config import config
from app.utils.logger import get_logger
from app.utils.retry import retry_on_exception
from app.pipeline.srt_utils import SubtitleSegment, SRTHandler

logger = get_logger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class SubtitleTranslator:
    """
    Handles translation of subtitle files using DeepL API.
    Preserves timestamps while translating text content.
    """

    # DeepL language code mapping
    LANGUAGE_CODES = {
        "ES": "ES",  # Spanish
        "DE": "DE",  # German
        "FR": "FR",  # French
        "IT": "IT",  # Italian
        "PT": "PT-PT",  # Portuguese
        "RU": "RU",  # Russian
        "JA": "JA",  # Japanese
        "ZH": "ZH",  # Chinese
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the subtitle translator.

        Args:
            api_key: DeepL API key (uses config if not provided)

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or config.DEEPL_API_KEY

        if not self.api_key:
            raise ValueError("DeepL API key is required")

        try:
            self.translator = deepl.Translator(self.api_key)
            logger.info("DeepL translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DeepL translator: {e}")
            raise TranslationError(f"Translator initialization failed: {e}")

    def check_usage(self) -> dict:
        """
        Check DeepL API usage statistics.

        Returns:
            Dictionary with usage information
        """
        try:
            usage = self.translator.get_usage()
            usage_dict = {
                "character_count": usage.character.count,
                "character_limit": usage.character.limit,
                "usage_percent": (usage.character.count / usage.character.limit * 100)
                if usage.character.limit else 0
            }

            logger.info(
                f"DeepL usage: {usage_dict['character_count']:,} / "
                f"{usage_dict['character_limit']:,} characters "
                f"({usage_dict['usage_percent']:.1f}%)"
            )

            return usage_dict

        except Exception as e:
            logger.warning(f"Could not fetch usage stats: {e}")
            return {}

    @retry_on_exception(exceptions=(TranslationError, deepl.DeepLException))
    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "EN"
    ) -> str:
        """
        Translate a single text string.

        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'ES', 'DE')
            source_lang: Source language code (default: 'EN')

        Returns:
            Translated text

        Raises:
            TranslationError: If translation fails
        """
        if not text.strip():
            return text

        try:
            # Map language code if needed
            deepl_target = self.LANGUAGE_CODES.get(target_lang.upper(), target_lang)

            result = self.translator.translate_text(
                text,
                source_lang=source_lang,
                target_lang=deepl_target
            )

            return result.text

        except deepl.DeepLException as e:
            logger.error(f"DeepL API error: {e}")
            raise TranslationError(f"Translation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            raise TranslationError(f"Translation failed: {e}")

    def translate_segments(
        self,
        segments: List[SubtitleSegment],
        target_lang: str,
        source_lang: str = "EN",
        batch_size: int = 50
    ) -> List[SubtitleSegment]:
        """
        Translate a list of subtitle segments.
        Preserves timestamps and indices.

        Args:
            segments: List of subtitle segments to translate
            target_lang: Target language code
            source_lang: Source language code (default: 'EN')
            batch_size: Number of segments to translate in one API call

        Returns:
            List of translated subtitle segments
        """
        logger.info(
            f"Translating {len(segments)} segments from {source_lang} to {target_lang}"
        )

        translated_segments: List[SubtitleSegment] = []

        # Process in batches to optimize API calls
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]

            # Extract texts from batch
            texts = [seg.text for seg in batch]

            try:
                # Translate batch
                deepl_target = self.LANGUAGE_CODES.get(
                    target_lang.upper(),
                    target_lang
                )

                results = self.translator.translate_text(
                    texts,
                    source_lang=source_lang,
                    target_lang=deepl_target
                )

                # Handle single vs multiple results
                if not isinstance(results, list):
                    results = [results]

                # Create translated segments with preserved timestamps
                for segment, result in zip(batch, results):
                    translated_segment = SubtitleSegment(
                        index=segment.index,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=result.text
                    )
                    translated_segments.append(translated_segment)

                logger.debug(
                    f"Translated batch {i // batch_size + 1}: "
                    f"{len(batch)} segments"
                )

            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # Fallback to individual translation
                logger.info("Falling back to individual segment translation...")

                for segment in batch:
                    try:
                        translated_text = self.translate_text(
                            segment.text,
                            target_lang,
                            source_lang
                        )

                        translated_segment = SubtitleSegment(
                            index=segment.index,
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            text=translated_text
                        )
                        translated_segments.append(translated_segment)

                    except Exception as segment_error:
                        logger.error(
                            f"Failed to translate segment {segment.index}: {segment_error}"
                        )
                        # Keep original text if translation fails
                        translated_segments.append(segment)

        logger.info(f"Translation complete: {len(translated_segments)} segments")
        return translated_segments

    def translate_srt_file(
        self,
        input_path: Path,
        output_path: Path,
        target_lang: str,
        source_lang: str = "EN"
    ) -> Path:
        """
        Translate an entire SRT file to a target language.

        Args:
            input_path: Path to source SRT file
            output_path: Path for translated SRT file
            target_lang: Target language code
            source_lang: Source language code (default: 'EN')

        Returns:
            Path to the translated SRT file
        """
        logger.info(
            f"Translating SRT file: {input_path.name} -> "
            f"{output_path.name} ({source_lang} to {target_lang})"
        )

        # Parse source SRT
        segments = SRTHandler.parse_srt(input_path)

        # Translate segments
        translated_segments = self.translate_segments(
            segments,
            target_lang,
            source_lang
        )

        # Write translated SRT
        SRTHandler.write_srt(translated_segments, output_path)

        logger.info(f"Translated SRT saved: {output_path}")
        return output_path


# Module-level convenience function
def translate_srt(
    input_path: Path,
    output_path: Path,
    target_lang: str,
    source_lang: str = "EN",
    api_key: Optional[str] = None
) -> Path:
    """
    Convenience function to translate an SRT file.

    Args:
        input_path: Path to source SRT file
        output_path: Path for translated SRT file
        target_lang: Target language code
        source_lang: Source language code (default: 'EN')
        api_key: Optional DeepL API key

    Returns:
        Path to the translated SRT file
    """
    translator = SubtitleTranslator(api_key=api_key)
    return translator.translate_srt_file(input_path, output_path, target_lang, source_lang)
