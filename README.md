# 🎥 Video Translation & Dubbing Pipeline with Lip Sync

Production-ready Python pipeline for automatically translating videos with full dubbing and lip synchronization. Transforms English videos into multiple languages with natural-sounding voiceovers and synchronized lip movements.

## ✨ Features

- **🎤 Audio Transcription**: Accurate speech-to-text using OpenAI Whisper
- **🌍 Translation**: High-quality subtitle translation via DeepL API
- **🗣️ Voice Dubbing**: Natural voice synthesis with ElevenLabs TTS
- **💋 Lip Synchronization**: Realistic lip sync using Video Retalking
- **⚡ Parallel Processing**: Multi-worker architecture for batch processing
- **📝 Subtitle Generation**: SRT files for all target languages
- **🎬 Complete Videos**: Final output with dubbed audio and synchronized lips

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16+ GB VRAM recommended)
- FFmpeg installed on your system

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/translate-v.git
cd translate-v
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

4. **Install Video Retalking (for lip sync)**
```bash
# Clone Video Retalking
git clone https://github.com/OpenTalker/video-retalking models/video-retalking
cd models/video-retalking

# Install dependencies
pip install -r requirements.txt

# Download checkpoints (follow repo instructions)
# https://github.com/OpenTalker/video-retalking#installation

cd ../..
```

5. **Configure environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required API keys:
- **DeepL API**: Get from [deepl.com/pro-api](https://www.deepl.com/pro-api)
- **ElevenLabs API**: Get from [elevenlabs.io](https://elevenlabs.io)

### Basic Usage

1. **Place videos in input directory**
```bash
# Create input directory (if not exists)
mkdir -p app/storage/input

# Copy your videos
cp /path/to/your/video.mp4 app/storage/input/
```

2. **Run the pipeline**
```bash
# Process all videos with default settings (Spanish + German)
python -m app.main

# Process with specific languages
python -m app.main --languages ES DE FR

# Use more workers for faster processing
python -m app.main --workers 4

# Custom input/output directories
python -m app.main --input /path/to/videos --output /path/to/output
```

3. **Check results**
```bash
# Output structure:
app/storage/output/
├── video_name/
│   ├── original_en.srt              # English subtitles
│   ├── spanish_es.srt               # Spanish subtitles
│   ├── german_de.srt                # German subtitles
│   ├── dubbed_spanish_es.mp3        # Spanish audio
│   ├── dubbed_german_de.mp3         # German audio
│   ├── final_spanish_es.mp4         # Spanish video with lip sync
│   └── final_german_de.mp4          # German video with lip sync
```

## 📋 Configuration

### Pipeline Modes

Configure what outputs to generate in `.env`:

#### Mode 1: Full Production (Subtitles + Dubbing + Lip Sync)
```bash
GENERATE_SUBTITLES=true
GENERATE_DUBBING=true
APPLY_LIPSYNC=true
```
- **Time**: ~30-60 min per hour of video
- **Cost**: $3-10 per video (ElevenLabs)
- **Output**: SRT files, dubbed audio, lip-synced videos

#### Mode 2: Subtitles Only
```bash
GENERATE_SUBTITLES=true
GENERATE_DUBBING=false
APPLY_LIPSYNC=false
```
- **Time**: ~5-10 min per hour of video
- **Cost**: Free (DeepL free tier)
- **Output**: SRT files only

#### Mode 3: Dubbing without Lip Sync
```bash
GENERATE_SUBTITLES=true
GENERATE_DUBBING=true
APPLY_LIPSYNC=false
```
- **Time**: ~10-15 min per hour of video
- **Cost**: $3-10 per video
- **Output**: SRT files, dubbed audio, videos with replaced audio

### Supported Languages

Currently configured for:
- **Spanish (ES)**
- **German (DE)**

To add more languages, edit `app/config.py`:
```python
TARGET_LANGUAGES: dict[str, str] = {
    "ES": "spanish",
    "DE": "german",
    "FR": "french",    # Add this
    "IT": "italian",   # Add this
}
```

## 🏗️ Architecture

### Project Structure
```
translate-v/
├── app/
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration management
│   ├── pipeline/               # Core processing modules
│   │   ├── extract_audio.py    # FFmpeg audio extraction
│   │   ├── transcribe.py       # Whisper transcription
│   │   ├── translate.py        # DeepL translation
│   │   ├── srt_utils.py        # SRT file handling
│   │   ├── tts_elevenlabs.py   # ElevenLabs voice synthesis
│   │   ├── lipsync_videoretalking.py  # Lip synchronization
│   │   ├── video_composer.py   # Video/audio composition
│   │   └── process_video.py    # Pipeline orchestration
│   ├── workers/                # Parallel processing
│   │   ├── queue.py            # Task queue management
│   │   └── worker.py           # Worker processes
│   └── utils/                  # Utilities
│       ├── logger.py           # Logging setup
│       └── retry.py            # Retry logic
├── models/                     # Lip sync models
│   ├── video-retalking/        # Video Retalking installation
│   └── Wav2Lip/                # Wav2Lip (alternative)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

### Processing Pipeline

```
┌──────────────┐
│ Input Video  │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ 1. Extract Audio    │  FFmpeg
│    (16kHz mono WAV) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 2. Transcribe       │  Whisper
│    (English SRT)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 3. Translate        │  DeepL
│    (Target SRTs)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 4. Generate Speech  │  ElevenLabs TTS
│    (Dubbed audio)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 5. Apply Lip Sync   │  Video Retalking
│    (Final video)    │
└──────┬──────────────┘
       │
       ▼
┌──────────────┐
│ Output Files │
└──────────────┘
```

## 💻 System Requirements

### Minimum Requirements (Subtitles Only)
- **CPU**: 4+ cores
- **RAM**: 8 GB
- **GPU**: Not required
- **Storage**: 10 GB free space

### Recommended (Full Pipeline with Lip Sync)
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16+ GB
- **GPU**: NVIDIA RTX 3080/3090/4080/4090 (16+ GB VRAM)
- **Storage**: 50+ GB free space

### GPU VRAM Requirements

| Configuration | VRAM Needed |
|---------------|-------------|
| Whisper tiny + Wav2Lip | 4-6 GB |
| Whisper base + Wav2Lip | 4-6 GB |
| Whisper medium + Video Retalking | 12-14 GB |
| Whisper large-v3 + Video Retalking | 16-18 GB |

## ⏱️ Performance

### Processing Time (1-hour video)

#### With GPU (RTX 3090)
| Step | Time |
|------|------|
| Audio Extraction | 10-30 sec |
| Transcription (Whisper large-v3) | 5-10 min |
| Translation | 30-60 sec |
| TTS (ElevenLabs) | 2-5 min |
| Lip Sync (Video Retalking) | 20-40 min |
| **Total** | **~30-60 min** |

#### With CPU Only
| Step | Time |
|------|------|
| Audio Extraction | 10-30 sec |
| Transcription | 60-120 min |
| Translation | 30-60 sec |
| TTS (ElevenLabs) | 2-5 min |
| Lip Sync | Not recommended |
| **Total** | **~90+ min (without lip sync)** |

## 💰 API Costs

### Per 1-Hour Video

**ElevenLabs TTS**:
- Characters needed: ~30,000-50,000 per language
- Cost: $3-10 per video (depending on plan)

**DeepL Translation**:
- Free tier: 500,000 characters/month
- Approximately 16-50 videos/month free
- Pro: Unlimited for $6.49/month

### Monthly Cost Estimates

| Videos/Month | ElevenLabs | DeepL | Total |
|--------------|------------|-------|-------|
| 10 videos | $30-100 | Free | $30-100 |
| 50 videos | $150-500 | $6.49 | $156-506 |
| 100 videos | $300-1000 | $6.49 | $306-1006 |

## 🛠️ CLI Reference

### Command Options

```bash
python -m app.main [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input directory with videos | `app/storage/input` |
| `--output, -o` | Output directory for results | `app/storage/output` |
| `--workers, -w` | Number of parallel workers | `4` |
| `--languages, -l` | Target languages (space-separated) | `ES DE` |
| `--model, -m` | Whisper model name | `large-v3` |
| `--save-results` | Save results to JSON file | None |
| `--dry-run` | List videos without processing | False |

### Examples

```bash
# Basic usage
python -m app.main

# Custom languages
python -m app.main --languages ES FR IT

# More workers for batch processing
python -m app.main --workers 8

# Faster model for testing
python -m app.main --model base --workers 1

# Save processing results
python -m app.main --save-results results.json

# Preview what will be processed
python -m app.main --dry-run
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (CUDA)
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use smaller Whisper model: `WHISPER_MODEL=base`
- Reduce workers: `MAX_WORKERS=1`
- Use Wav2Lip instead of Video Retalking: `LIPSYNC_METHOD=wav2lip`

#### 2. FFmpeg Not Found
```
FileNotFoundError: ffmpeg not found
```
**Solution**: Install FFmpeg for your system (see Installation section)

#### 3. Video Retalking Not Found
```
Video Retalking not found at ./models/video-retalking
```
**Solution**: Follow Video Retalking installation steps or disable lip sync:
```bash
APPLY_LIPSYNC=false
```

#### 4. API Rate Limits
```
ElevenLabs API error: 429 Too Many Requests
```
**Solution**: Reduce concurrent workers or upgrade API plan

### Debug Mode

Enable detailed logging:
```bash
# In .env
LOG_LEVEL=DEBUG

# Run with verbose output
python -m app.main 2>&1 | tee debug.log
```

## 📊 Benchmarks

Tested on various hardware configurations:

| GPU | VRAM | Time (1hr video) | Cost |
|-----|------|------------------|------|
| RTX 4090 | 24 GB | 25-35 min | Optimal |
| RTX 3090 | 24 GB | 30-45 min | Excellent |
| RTX 3080 | 10 GB | 45-60 min | Good (base Whisper) |
| RTX 3060 | 12 GB | 50-70 min | Acceptable |
| CPU (i9-12900K) | N/A | 120+ min | Slow (no lip sync) |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Speech recognition
- **DeepL**: Translation API
- **ElevenLabs**: Voice synthesis
- **Video Retalking**: Lip synchronization
- **FFmpeg**: Media processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/translate-v/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/translate-v/discussions)

---

Made with ❤️ for the video localization community
