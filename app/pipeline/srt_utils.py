"""
SRT (SubRip) file format utilities.
Handles parsing, creating, and manipulating subtitle files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SubtitleSegment:
    """
    Represents a single subtitle segment with timing and text.
    """
    index: int
    start_time: str  # Format: HH:MM:SS,mmm
    end_time: str    # Format: HH:MM:SS,mmm
    text: str

    def __str__(self) -> str:
        """
        Convert segment to SRT format string.

        Returns:
            Formatted SRT segment
        """
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"

    def to_dict(self) -> dict:
        """Convert segment to dictionary format."""
        return {
            "index": self.index,
            "start": self.start_time,
            "end": self.end_time,
            "text": self.text
        }


class SRTHandler:
    """
    Handles reading, writing, and manipulating SRT subtitle files.
    """

    # SRT timestamp format regex: HH:MM:SS,mmm
    TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    )

    @staticmethod
    def parse_srt(file_path: Path) -> List[SubtitleSegment]:
        """
        Parse an SRT file into a list of SubtitleSegment objects.

        Args:
            file_path: Path to SRT file

        Returns:
            List of subtitle segments

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"SRT file not found: {file_path}")

        segments: List[SubtitleSegment] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by double newline (segment separator)
        raw_segments = content.strip().split('\n\n')

        for raw_segment in raw_segments:
            lines = raw_segment.strip().split('\n')

            if len(lines) < 3:
                logger.warning(f"Skipping malformed segment: {raw_segment[:50]}")
                continue

            try:
                # Parse index
                index = int(lines[0])

                # Parse timestamps
                timestamp_line = lines[1]
                match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    timestamp_line
                )

                if not match:
                    logger.warning(f"Invalid timestamp format: {timestamp_line}")
                    continue

                start_time, end_time = match.groups()

                # Parse text (can be multiple lines)
                text = '\n'.join(lines[2:])

                segment = SubtitleSegment(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                )
                segments.append(segment)

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing segment: {e}")
                continue

        logger.info(f"Parsed {len(segments)} segments from {file_path.name}")
        return segments

    @staticmethod
    def write_srt(segments: List[SubtitleSegment], output_path: Path) -> None:
        """
        Write subtitle segments to an SRT file.

        Args:
            segments: List of subtitle segments
            output_path: Path where SRT file will be saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(str(segment))
                f.write('\n')  # Extra newline between segments

        logger.info(f"Wrote {len(segments)} segments to {output_path.name}")

    @staticmethod
    def create_segment(
        index: int,
        start_time: str,
        end_time: str,
        text: str
    ) -> SubtitleSegment:
        """
        Create a new subtitle segment.

        Args:
            index: Segment number (1-based)
            start_time: Start timestamp (HH:MM:SS,mmm)
            end_time: End timestamp (HH:MM:SS,mmm)
            text: Subtitle text

        Returns:
            New SubtitleSegment object
        """
        return SubtitleSegment(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text
        )

    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """
        Validate SRT timestamp format.

        Args:
            timestamp: Timestamp string to validate

        Returns:
            True if valid, False otherwise
        """
        return bool(SRTHandler.TIMESTAMP_PATTERN.match(timestamp))

    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        """
        Convert SRT timestamp to seconds.

        Args:
            timestamp: Timestamp in HH:MM:SS,mmm format

        Returns:
            Time in seconds
        """
        match = SRTHandler.TIMESTAMP_PATTERN.match(timestamp)
        if not match:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        hours, minutes, seconds, milliseconds = map(int, match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return total_seconds

    @staticmethod
    def seconds_to_timestamp(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format.

        Args:
            seconds: Time in seconds

        Returns:
            Timestamp in HH:MM:SS,mmm format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def merge_segments(segments: List[SubtitleSegment], max_gap: float = 1.0) -> List[SubtitleSegment]:
        """
        Merge consecutive segments with small gaps between them.

        Args:
            segments: List of subtitle segments
            max_gap: Maximum gap in seconds to merge (default 1.0)

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged: List[SubtitleSegment] = []
        current = segments[0]

        for next_segment in segments[1:]:
            current_end = SRTHandler.timestamp_to_seconds(current.end_time)
            next_start = SRTHandler.timestamp_to_seconds(next_segment.start_time)

            gap = next_start - current_end

            if gap <= max_gap:
                # Merge segments
                current = SubtitleSegment(
                    index=current.index,
                    start_time=current.start_time,
                    end_time=next_segment.end_time,
                    text=f"{current.text} {next_segment.text}"
                )
            else:
                merged.append(current)
                current = next_segment

        merged.append(current)

        # Re-index merged segments
        for i, segment in enumerate(merged, 1):
            segment.index = i

        return merged
