# ⚡ RaijinCut — Parallel Video Processing CLI

> *Named after Raijin, the Japanese god of thunder — lightning splits into multiple bolts, just like RaijinCut splits video work across parallel threads.*

A hybrid Rust + Python CLI tool for video transcription, repurposing, and editing. Wraps FFmpeg, yt-dlp, and Python-based ML tools into a single, thunder-fast binary.

## Why Hybrid? (Rust + Python)

Most video CLI tools are written entirely in Python. RaijinCut takes a different approach — a **compiled Rust binary** orchestrates everything, calling Python only for ML/AI tasks where the Python ecosystem is essential.

### What this gets you

| Concern | Pure Python tools | RaijinCut (hybrid) |
|---|---|---|
| **CLI startup** | ~100ms+ interpreter overhead | Near-instant (compiled binary) |
| **Parallel batch jobs** | GIL-limited; needs `multiprocessing` | OS-level threads, true parallelism |
| **Distribution** | `pip install` + virtualenv + dependency hell | Single binary + Python for ML only |
| **ML/AI features** | Native access | Full access (shells out to Python) |
| **Adding new ffmpeg commands** | Slow subprocess wrapper in Python | Zero-cost `Command::new()` in Rust |
| **Memory safety** | Runtime errors | Compile-time guarantees |

### Two levels of parallelism

1. **Across videos** — `batch-transcribe` processes N videos simultaneously using Rust's `std::thread` pool. Each thread spawns an independent Python subprocess, completely bypassing Python's GIL.
2. **Within a single video** — the `transcribe` command runs Whisper transcription and speaker diarization as parallel threads, then merges results. This cuts transcription time nearly in half for diarized output.

A pure-Python tool using `multiprocessing` can achieve similar parallelism, but with more overhead (process pickling, no shared memory) and significantly more boilerplate. RaijinCut's Rust layer makes parallelism trivial.

## Features

| Command | Description |
|---------|-------------|
| `transcribe` | Transcribe video to text with timestamps (OpenAI Whisper) |
| `batch-transcribe` | Transcribe multiple videos in parallel from a URL list |
| `probe` | Get video metadata (resolution, duration, codec, audio) |
| `download` | Download video from YouTube/URL via yt-dlp (max 1080p) |
| `cut` | Cut video segment by start/end timestamps (stream copy, instant) |
| `crop` | Crop to aspect ratio (9:16, 16:9, 1:1, 4:5) with auto-centering |
| `subtitle` | Burn SRT subtitles into video (styled, positioned) |
| `convert` | Re-encode video with configurable CRF quality |
| `pipeline` | Full end-to-end: download → transcribe → AI highlights → cut → filler removal → stitch → crop → subtitle |
| `repurpose` | Simpler pipeline: download → cut → crop → scale → subtitle |
| `rough-cut` | Remove filler word segments from video using SRT timing |
| `presets` | List available platform presets |

## Quick Start

### Build

```bash
cargo build --release
# Binary at target/release/raijincut
```

### Requirements

| Dependency | Purpose | Install |
|---|---|---|
| **FFmpeg** + **ffprobe** | All video/audio processing | `brew install ffmpeg` / `apt install ffmpeg` |
| **yt-dlp** | Video downloading | `brew install yt-dlp` / `pip install yt-dlp` |
| **Python 3.x** | ML scripts (transcription, subtitles, rough-cut) | Usually pre-installed |
| **moviepy**, **pysubs2** | Subtitle burning, rough-cut editing | `pip install moviepy pysubs2` |
| **OpenAI API key** | Whisper transcription | Set `OPENAI_API_KEY` env var |
| **HF_TOKEN** (optional) | Speaker diarization (pyannote) | Set `HF_TOKEN` env var |

```bash
pip install moviepy pysubs2 pillow openai
```

## Usage

### Probe video metadata

```bash
raijincut probe video.mp4
# Output: File, Duration, Video resolution, Audio codec
```

### Download from YouTube

```bash
raijincut download "https://youtube.com/watch?v=..." -o ./downloads
```

### Transcribe with timestamps

```bash
# Basic transcription
raijincut transcribe "https://youtube.com/watch?v=..." -o transcript.md

# With speaker diarization (runs transcription + diarization in parallel)
raijincut transcribe "https://youtube.com/watch?v=..." -o transcript.md -d
```

### Batch transcribe (parallel)

```bash
# Create a file with one URL per line
cat > urls.txt << 'EOF'
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3
https://youtube.com/watch?v=VIDEO4
EOF

# Process 4 videos concurrently
raijincut batch-transcribe urls.txt -o ./transcripts -j 4

# With speaker diarization on all videos
raijincut batch-transcribe urls.txt -o ./transcripts -d -j 4
```

Lines starting with `#` are ignored, so you can comment out URLs in the list file.

### Cut, crop, convert

```bash
# Cut a segment (start to end timestamp, stream copy = instant)
raijincut cut video.mp4 -s 00:00:30 -e 00:02:00 -o clip.mp4

# Crop to vertical (9:16) — auto-centers the crop region
raijincut crop video.mp4 -o vertical.mp4 -a 9:16

# Re-encode with quality setting (CRF 0–51, lower = better)
raijincut convert input.mp4 -o output.mp4 -c 18
```

### Burn subtitles

```bash
raijincut subtitle video.mp4 --srt subs.srt -o final.mp4
```

Subtitles are rendered with white text, black stroke, positioned at the bottom of the frame.

### Remove filler words

```bash
raijincut rough-cut video.mp4 --srt transcript.srt -o clean.mp4
```

Detects and removes segments where every word is a filler (um, uh, er, ah, like, basically, actually, literally, honestly, okay, right, I mean). Adjacent segments are merged with small padding to avoid jarring cuts.

### Pipeline — AI-powered highlight reels

The `pipeline` command is the full end-to-end workflow. Give it a YouTube URL and a preset, and it will:

1. Download the video
2. Transcribe + diarize (in parallel)
3. **AI highlight selection** — sends transcript to GPT-4o-mini to pick 3-5 compelling segments (60-90s total)
4. Cut highlight segments (stream copy, instant)
5. Remove filler words from each clip
6. Stitch clips with transitions (crossfade, wipeleft, etc.)
7. Crop + scale to platform format
8. Burn styled subtitles + fade effects

```bash
# Create a 60-90s highlight reel for Instagram Reels
raijincut pipeline "https://youtube.com/watch?v=..." -p reels -o ./output

# With a specific transition style
raijincut pipeline "https://youtube.com/watch?v=..." -p reels -t crossfade -o ./output

# Process multiple videos
raijincut pipeline "URL1" "URL2" -p tiktok -o ./output
```

If the OpenAI API key is not set, highlight selection falls back to keyword-based scoring (no AI required).

### Repurpose for social media

Simpler pipeline — downloads the video, cuts a clip, crops to the platform's aspect ratio, scales to target resolution, and optionally burns subtitles:

```bash
# Repurpose a YouTube video for TikTok with subtitles
raijincut repurpose "https://youtube.com/watch?v=..." \
  -p tiktok -c 00:00:30-00:02:00 --subtitles -o tiktok_clip.mp4

# Repurpose for Instagram Reels (no subtitles)
raijincut repurpose "https://youtube.com/watch?v=..." \
  -p reels -c 00:01:00-00:02:00 -o reel.mp4

# See all available presets
raijincut presets
```

## Platform Presets

Presets are TOML files in `presets/` that define resolution, aspect ratio, codec, and duration limits per platform:

| Preset | Resolution | Aspect | FPS | Max Duration |
|--------|-----------|--------|-----|-------------|
| tiktok | 1080x1920 | 9:16 | 30 | 180s |
| reels | 1080x1920 | 9:16 | 30 | 90s |
| shorts | 1080x1920 | 9:16 | 30 | 60s |
| linkedin | 1080x1350 | 4:5 | 30 | — |
| twitter | 1280x720 | 16:9 | 30 | 140s |
| square | 1080x1080 | 1:1 | 30 | — |

You can add custom presets by creating a new TOML file in `presets/`.

## Architecture

```
raijincut/
├── src/main.rs              # Rust CLI — argument parsing, process orchestration,
│                             #   threading, and all video processing commands
├── python/
│   ├── transcribe_only.py   # Whisper transcription (OpenAI API)
│   ├── diarize_only.py      # Speaker diarization (pyannote)
│   ├── merge_transcript.py  # Merge transcription + diarization output
│   ├── video_transcribe.py  # Full transcription pipeline with chunking
│   ├── highlight_selector.py# AI highlight selection (keyword + GPT-4o-mini)
│   ├── video_repurpose.py   # Advanced repurposing pipeline
│   ├── filler_remover.py    # Audio + transcript filler analysis
│   ├── ai_filler_decider.py # AI-powered filler decisions (OpenAI)
│   ├── transcript_filler.py # Transcript filler word detection
│   ├── speaker_detector.py  # Speaker diarization (pyannote/whisperx)
│   ├── audio_analysis.py    # Silence detection
│   ├── video_format.py      # Format conversion utilities
│   ├── video_transitions.py # Transition effects (xfade)
│   └── fetch_metadata.py    # Video metadata fetching
├── presets/                  # Platform preset configs (TOML)
│   ├── tiktok.toml
│   ├── reels.toml
│   ├── shorts.toml
│   ├── linkedin.toml
│   ├── twitter.toml
│   ├── square.toml
│   └── default.toml
├── Cargo.toml               # Rust dependencies (clap, serde, ort, image, etc.)
└── CLAUDE.md                # AI assistant instructions
```

### How it works

The Rust binary is the single entry point. It uses [clap](https://docs.rs/clap) for argument parsing and dispatches to handler functions that:

1. **For simple operations** (cut, crop, convert, probe, download) — directly invoke `ffmpeg`/`ffprobe`/`yt-dlp` via `std::process::Command`
2. **For ML operations** (transcribe, rough-cut, subtitle) — shell out to Python scripts or embed Python code inline
3. **For pipelines** (pipeline, repurpose) — chain multiple steps sequentially, with the `pipeline` command adding AI highlight selection and multi-clip stitching with transitions
4. **For batch work** (batch-transcribe) — spawn OS threads that each run independent processing pipelines

The `ort` crate (ONNX Runtime) is included in `Cargo.toml` for planned on-device ML inference, which would reduce the Python dependency for some features.

## Standalone Python Scripts

The Python scripts in `python/` can also be used independently for more control:

```bash
# Transcribe with chunking for long videos
python3 python/video_transcribe.py "URL" -o ./transcripts

# Full repurpose with auto-highlight detection
python3 python/video_repurpose.py "URL" --auto --subtitles --format reels

# Speaker diarization (requires HF_TOKEN)
python3 python/speaker_detector.py video.mp4
```

## License

MIT
