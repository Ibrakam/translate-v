"""
Simple test script to process a single video with local lip sync.
"""
from pathlib import Path
from app.pipeline.process_video import VideoProcessor
from app.pipeline.lipsync_videoretalking import VideoRetalkingLipSync

# Path to your video (using short version for testing)
video_path = Path("app/storage/input/IMG_1752_short.mp4")

# Process with Spanish only for faster testing
print("Starting video processing with LOCAL lip sync...")
print(f"Video: {video_path}")
print("=" * 70)

# Initialize processor with local Video Retalking lip sync
print("\n🖥️  Using LOCAL Video Retalking for lip sync")
lipsync = VideoRetalkingLipSync()
processor = VideoProcessor(lipsync=lipsync)

result = processor.process_video(video_path, target_languages=["ES"])

print("\n" + "=" * 70)
if result.success:
    print("✓ SUCCESS!")
    print(f"Processing time: {result.processing_time:.2f}s ({result.processing_time/60:.2f} min)")
    print(f"\nGenerated files:")
    print(f"  English SRT: {result.english_srt}")
    if result.translated_srts:
        for lang, path in result.translated_srts.items():
            print(f"  {lang} SRT: {path}")
    if result.dubbed_audios:
        for lang, path in result.dubbed_audios.items():
            print(f"  {lang} Audio: {path}")
    if result.dubbed_videos:
        for lang, path in result.dubbed_videos.items():
            print(f"  {lang} Video (with lip sync): {path}")
else:
    print("✗ FAILED!")
    print(f"Error: {result.error}")
print("=" * 70)
