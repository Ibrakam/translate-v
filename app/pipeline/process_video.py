"""
Video processing orchestrator for full dubbing with lip sync.
Coordinates: extraction, transcription, translation, TTS, and lip sync.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time

from app.config import config
from app.utils.logger import get_logger
from app.pipeline.extract_audio import AudioExtractor
from app.pipeline.transcribe import WhisperTranscriber
from app.pipeline.translate import SubtitleTranslator
from app.pipeline.tts_elevenlabs import ElevenLabsTTS
from app.pipeline.tts_edge import EdgeTTS
from app.pipeline.lipsync_videoretalking import VideoRetalkingLipSync
from app.pipeline.lipsync_replicate import ReplicateLipSync
from app.pipeline.lipsync_huggingface import HuggingFaceLipSync
from app.pipeline.video_composer import VideoComposer

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """
    Result of video processing pipeline.
    """
    video_name: str
    video_path: Path
    success: bool
    english_srt: Optional[Path] = None
    translated_srts: Optional[Dict[str, Path]] = None
    dubbed_videos: Optional[Dict[str, Path]] = None
    dubbed_audios: Optional[Dict[str, Path]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "video_name": self.video_name,
            "video_path": str(self.video_path),
            "success": self.success,
            "english_srt": str(self.english_srt) if self.english_srt else None,
            "translated_srts": {
                lang: str(path) for lang, path in (self.translated_srts or {}).items()
            },
            "dubbed_videos": {
                lang: str(path) for lang, path in (self.dubbed_videos or {}).items()
            },
            "dubbed_audios": {
                lang: str(path) for lang, path in (self.dubbed_audios or {}).items()
            },
            "error": self.error,
            "processing_time": self.processing_time
        }


class VideoProcessor:
    """
    Orchestrates the complete video dubbing pipeline with lip sync.
    Handles: extraction, transcription, translation, TTS, and lip synchronization.
    """

    def __init__(
        self,
        transcriber: Optional[WhisperTranscriber] = None,
        translator: Optional[SubtitleTranslator] = None,
        tts: Optional[ElevenLabsTTS] = None,
        lipsync: Optional[VideoRetalkingLipSync] = None,
        cleanup_temp_files: bool = True
    ):
        """
        Initialize the video processor.

        Args:
            transcriber: Optional pre-initialized WhisperTranscriber
            translator: Optional pre-initialized SubtitleTranslator
            tts: Optional pre-initialized ElevenLabsTTS
            lipsync: Optional pre-initialized VideoRetalkingLipSync
            cleanup_temp_files: Whether to remove temporary files
        """
        self.audio_extractor = AudioExtractor()
        self.transcriber = transcriber or WhisperTranscriber()
        self.video_composer = VideoComposer()

        # Initialize optional components based on config
        if config.GENERATE_SUBTITLES:
            self.translator = translator or SubtitleTranslator()
        else:
            self.translator = None

        if config.GENERATE_DUBBING:
            if tts:
                self.tts = tts
            elif config.TTS_PROVIDER == "elevenlabs":
                self.tts = ElevenLabsTTS()
            else:  # default to edge (free)
                self.tts = EdgeTTS()
        else:
            self.tts = None

        if config.APPLY_LIPSYNC:
            if lipsync:
                self.lipsync = lipsync
            elif config.LIPSYNC_METHOD == "huggingface":
                self.lipsync = HuggingFaceLipSync()
            elif config.LIPSYNC_METHOD == "replicate":
                self.lipsync = ReplicateLipSync()
            else:  # default to videoretalking (local)
                self.lipsync = VideoRetalkingLipSync()
        else:
            self.lipsync = None

        self.cleanup_temp_files = cleanup_temp_files

        logger.info("Video processor initialized")
        logger.info(f"  Subtitles: {config.GENERATE_SUBTITLES}")
        logger.info(f"  Dubbing: {config.GENERATE_DUBBING}")
        logger.info(f"  Lip Sync: {config.APPLY_LIPSYNC}")

    def process_video(
        self,
        video_path: Path,
        target_languages: Optional[List[str]] = None
    ) -> ProcessingResult:
        """
        Process a single video through the complete pipeline.

        Args:
            video_path: Path to video file
            target_languages: List of language codes to translate to (default: ES, DE)

        Returns:
            ProcessingResult with paths to generated files
        """
        start_time = datetime.now()
        video_name = video_path.stem

        logger.info(f"=" * 70)
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"=" * 70)

        # Use default target languages if not specified
        if target_languages is None:
            target_languages = list(config.TARGET_LANGUAGES.keys())

        # Prepare output directory
        output_dir = config.get_output_path(video_name)
        audio_path = None
        temp_files = []

        try:
            # ===================================================================
            # STEP 1: Extract audio from video
            # ===================================================================
            logger.info("[1/5] Extracting audio from video...")
            step_start = time.time()
            audio_path = self.audio_extractor.extract_audio(video_path)
            temp_files.append(audio_path)
            step_time = time.time() - step_start
            logger.info(f"  ⏱️  Audio extraction took: {step_time:.2f}s")

            # ===================================================================
            # STEP 2: Transcribe audio to English SRT
            # ===================================================================
            logger.info("[2/5] Transcribing audio to English subtitles...")
            step_start = time.time()
            english_srt_path = output_dir / "original_en.srt"

            self.transcriber.transcribe_and_save_srt(
                audio_path,
                english_srt_path,
                language="en"
            )
            step_time = time.time() - step_start
            logger.info(f"  ⏱️  Transcription took: {step_time:.2f}s ({step_time/60:.2f} min)")

            # ===================================================================
            # STEP 3: Translate subtitles (if enabled)
            # ===================================================================
            translated_srts: Dict[str, Path] = {}

            if config.GENERATE_SUBTITLES:
                logger.info(f"[3/5] Translating subtitles to {len(target_languages)} languages...")
                translation_start = time.time()

                for lang_code in target_languages:
                    lang_name = config.TARGET_LANGUAGES.get(lang_code, lang_code.lower())
                    output_srt = output_dir / f"{lang_name}_{lang_code.lower()}.srt"

                    logger.info(f"  - Translating to {lang_code} ({lang_name})...")
                    lang_start = time.time()

                    self.translator.translate_srt_file(
                        input_path=english_srt_path,
                        output_path=output_srt,
                        target_lang=lang_code,
                        source_lang="EN"
                    )

                    translated_srts[lang_code] = output_srt
                    lang_time = time.time() - lang_start
                    logger.info(f"    ⏱️  {lang_code} translation took: {lang_time:.2f}s")

                total_translation_time = time.time() - translation_start
                logger.info(f"  ⏱️  Total translation time: {total_translation_time:.2f}s")
            else:
                logger.info("[3/5] Skipping subtitle translation (disabled in config)")

            # ===================================================================
            # STEP 4: Generate dubbed audio (if enabled)
            # ===================================================================
            dubbed_audios: Dict[str, Path] = {}

            if config.GENERATE_DUBBING:
                logger.info(f"[4/5] Generating dubbed audio for {len(target_languages)} languages...")
                tts_start = time.time()

                for lang_code in target_languages:
                    lang_name = config.TARGET_LANGUAGES.get(lang_code, lang_code.lower())

                    # Get translated SRT
                    srt_path = translated_srts.get(lang_code)
                    if not srt_path:
                        logger.warning(f"No SRT file for {lang_code}, skipping dubbing")
                        continue

                    # Generate dubbed audio
                    audio_output = output_dir / f"dubbed_{lang_name}_{lang_code.lower()}.mp3"

                    logger.info(f"  - Generating {lang_code} audio with ElevenLabs...")
                    lang_start = time.time()

                    self.tts.generate_full_audio(
                        srt_path=srt_path,
                        output_path=audio_output,
                        language=lang_code
                    )

                    dubbed_audios[lang_code] = audio_output
                    lang_time = time.time() - lang_start
                    logger.info(f"    ⏱️  {lang_code} TTS generation took: {lang_time:.2f}s")

                total_tts_time = time.time() - tts_start
                logger.info(f"  ⏱️  Total TTS time: {total_tts_time:.2f}s ({total_tts_time/60:.2f} min)")
            else:
                logger.info("[4/5] Skipping audio dubbing (disabled in config)")

            # ===================================================================
            # STEP 5: Apply lip sync (if enabled)
            # ===================================================================
            dubbed_videos: Dict[str, Path] = {}

            if config.APPLY_LIPSYNC and config.GENERATE_DUBBING:
                logger.info(f"[5/5] Applying lip sync for {len(target_languages)} languages...")
                lipsync_start = time.time()

                for lang_code in target_languages:
                    lang_name = config.TARGET_LANGUAGES.get(lang_code, lang_code.lower())

                    # Get dubbed audio
                    audio_output = dubbed_audios.get(lang_code)
                    if not audio_output:
                        logger.warning(f"No dubbed audio for {lang_code}, skipping lip sync")
                        continue

                    # Apply lip sync
                    video_output = output_dir / f"final_{lang_name}_{lang_code.lower()}.mp4"

                    logger.info(f"  - Applying lip sync for {lang_code}...")
                    lang_start = time.time()

                    try:
                        self.lipsync.apply_lipsync(
                            video_path=video_path,
                            audio_path=audio_output,
                            output_path=video_output
                        )

                        dubbed_videos[lang_code] = video_output

                    except Exception as e:
                        logger.error(f"Lip sync failed for {lang_code}, falling back to audio replacement: {e}")

                        # Fallback: just replace audio without lip sync
                        self.video_composer.replace_audio(
                            video_path=video_path,
                            audio_path=audio_output,
                            output_path=video_output
                        )

                        dubbed_videos[lang_code] = video_output

                    lang_time = time.time() - lang_start
                    logger.info(f"    ⏱️  {lang_code} lip sync took: {lang_time:.2f}s ({lang_time/60:.2f} min)")

                total_lipsync_time = time.time() - lipsync_start
                logger.info(f"  ⏱️  Total lip sync time: {total_lipsync_time:.2f}s ({total_lipsync_time/60:.2f} min)")

            elif config.GENERATE_DUBBING:
                # Dubbing enabled but lip sync disabled - just replace audio
                logger.info("[5/5] Replacing audio without lip sync...")

                for lang_code in target_languages:
                    lang_name = config.TARGET_LANGUAGES.get(lang_code, lang_code.lower())

                    audio_output = dubbed_audios.get(lang_code)
                    if not audio_output:
                        continue

                    video_output = output_dir / f"final_{lang_name}_{lang_code.lower()}.mp4"

                    self.video_composer.replace_audio(
                        video_path=video_path,
                        audio_path=audio_output,
                        output_path=video_output
                    )

                    dubbed_videos[lang_code] = video_output
            else:
                logger.info("[5/5] Skipping video dubbing (disabled in config)")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"✓ Video processing complete: {video_path.name}")
            logger.info(f"  Processing time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
            logger.info(f"  Output directory: {output_dir}")

            return ProcessingResult(
                video_name=video_name,
                video_path=video_path,
                success=True,
                english_srt=english_srt_path,
                translated_srts=translated_srts if translated_srts else None,
                dubbed_videos=dubbed_videos if dubbed_videos else None,
                dubbed_audios=dubbed_audios if dubbed_audios else None,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            logger.error(f"✗ Video processing failed: {video_path.name}")
            logger.error(f"  Error: {error_msg}")

            return ProcessingResult(
                video_name=video_name,
                video_path=video_path,
                success=False,
                error=error_msg,
                processing_time=processing_time
            )

        finally:
            # Cleanup temporary files
            if self.cleanup_temp_files:
                for temp_file in temp_files:
                    if temp_file and temp_file.exists():
                        try:
                            temp_file.unlink()
                            logger.debug(f"Cleaned up: {temp_file.name}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup {temp_file}: {e}")

    def process_multiple_videos(
        self,
        video_paths: List[Path],
        target_languages: Optional[List[str]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple videos sequentially.

        Args:
            video_paths: List of video file paths
            target_languages: List of language codes to translate to

        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Processing {len(video_paths)} videos sequentially...")

        results: List[ProcessingResult] = []

        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"\nVideo {i}/{len(video_paths)}")
            result = self.process_video(video_path, target_languages)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total videos: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        if failed > 0:
            logger.info("\nFailed videos:")
            for result in results:
                if not result.success:
                    logger.info(f"  - {result.video_name}: {result.error}")

        return results


def process_video_file(
    video_path: Path,
    target_languages: Optional[List[str]] = None
) -> ProcessingResult:
    """
    Convenience function to process a single video.

    Args:
        video_path: Path to video file
        target_languages: List of language codes to translate to

    Returns:
        ProcessingResult object
    """
    processor = VideoProcessor()
    return processor.process_video(video_path, target_languages)
