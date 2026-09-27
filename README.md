# ⚡ RaijinCut: Parallel Video Processing CLI

> *Named after Raijin, the Japanese god of thunder, lightning splits into multiple bolts, just like RaijinCut splits video work across parallel threads.*

A hybrid Rust + Python CLI tool for video transcription, repurposing, and editing. Wraps FFmpeg, yt-dlp, and Python-based ML tools into a single, thunder-fast binary.

## Why Hybrid? (Rust + Python)

Most video CLI tools are written entirely in Python. RaijinCut takes a different approach, a **compiled Rust binary** orchestrates everything, calling Python only for ML/AI tasks where the Python ecosystem is essential.

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

1. **Across videos**, `batch-transcribe` processes N videos simultaneously using Rust's `std::thread` pool. Each thread spawns an independent Python subprocess, completely bypassing Python's GIL.
2. **Within a single video**, the `transcribe` command runs Whisper transcription and speaker diarization as parallel threads, then merges results. This cuts transcription time nearly in half for diarized output.

A pure-Python tool using `multiprocessing` can achieve similar parallelism, but with more overhead (process pickling, no shared memory) and significantly more boilerplate. RaijinCut's Rust layer makes parallelism trivial.

## Features

| Command | Description |
|---------|-------------|
| `transcribe` | Transcribe video to text with timestamps (OpenAI Whisper) |
| `batch-transcribe` | Transcribe multiple videos in parallel from a URL list |
| `probe` | Get video metadata (resolution, duration, codec, audio) |
| `download` | Download a video (YouTube, X, Facebook, LinkedIn) as mp4 + metadata JSON; optional browser cookies |
| `analyze` | Break a video (file or URL) down: word-level transcript + SRT, shots and transitions, pacing, keyframes, contact sheet, vision notes, audio/SFX hits, caption style, `breakdown.json` + `breakdown.md` |
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
| **whisper.cpp** (optional) | Word-level transcript for `analyze` | `brew install whisper-cpp` + a ggml model |
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

### Download a video

```bash
raijincut download "https://youtube.com/shorts/..." -o ./downloads

# LinkedIn / X / Facebook posts usually need a signed-in session.
# Sign in to the site in Chrome yourself, then reuse that session:
raijincut download "https://www.linkedin.com/posts/..." -o ./downloads --cookies-from-browser chrome
```

Writes `<title>-<id>.mp4` (best mp4 with audio) and `<title>-<id>.json` next to it with `id`, `title`,
`uploader`, `channel`, `webpage_url`, `source_url`, `extractor_key`, `duration`, `view_count`, `like_count`,
`comment_count`, `upload_date`, `description` and `filepath` (counts are `null` when the site hides them).
raijincut never asks for, types or stores passwords. If a login wall blocks the download it prints a
message telling you to sign in in Chrome and rerun with `--cookies-from-browser chrome`.

### Analyze a reference video

```bash
# a local file, or a URL (downloaded first)
raijincut analyze ./downloads/video.mp4 -o ./breakdowns
raijincut analyze "https://youtube.com/shorts/..." -o ./breakdowns \
  --whisper-model ~/models/ggml-small.en.bin --vision auto
```

Everything lands in `<output>/<slug>/`:

| File | What |
|---|---|
| `breakdown.md` | Human summary: hook in the first 3 s, structure beats, cut cadence, transition mix, SFX usage, caption style, shot list, what to steal |
| `breakdown.json` | All data, stable schema below |
| `transcript.json`, `transcript.srt` | Word-level transcript and caption-sized SRT cues |
| `keyframes/shotNNN_{first,mid,last}.jpg` | Three frames per shot |
| `strips/shotNNN.jpg` | `[previous shot's last frame, first, mid, last]` in one image (what the vision pass sees) |
| `contact_sheet.jpg` | Mid frame of every shot with index, start time and length |

Options and backends:

| Section | How | Needs |
|---|---|---|
| Transcript | whisper.cpp (`whisper-cli -ml 1 -sow`) with `--whisper-model` / `RAIJINCUT_WHISPER_MODEL`; otherwise `mlx_whisper --word-timestamps` (`RAIJINCUT_MLX_MODEL` picks the model); otherwise skipped with a TODO | `whisper-cli` + a ggml model, or `mlx_whisper` |
| Shots | ffmpeg `scene` score spikes = hard cuts (`--scene-threshold`, default 0.3); luma runs = fade through black / white flash; pixel blend test on 32 px frames = dissolves | ffmpeg |
| Keyframes | ffmpeg; strips and contact sheet use Pillow (skipped without it) | ffmpeg, optional Pillow |
| Vision | `--vision auto` uses the Claude API when `ANTHROPIC_API_KEY` is set (`pip install anthropic`, model `RAIJINCUT_VISION_MODEL`, default `claude-opus-5`), else a local ollama vision model (`RAIJINCUT_OLLAMA_MODEL`), else skips with a TODO. `--vision claude-cli` uses the local `claude` CLI (your Claude Code login) instead. Keys come from the environment only | one of the above |
| Audio | ffmpeg `ebur128` loudness; 10 ms envelopes in three bands (full, >3 kHz, <150 Hz) for SFX hits; noise floor vs median for a music bed guess | ffmpeg |
| Captions | Burned-in caption style aggregated from the vision pass | a vision backend |
| Summary | Heuristic hook/beats/what-to-steal always; with `api` or `claude-cli` also an LLM pass for beats, format formula and what to steal (`--no-summary` in `python/analyze.py` skips it) | optional |

Only stdlib Python is required (plus optional Pillow and `anthropic`). Run `python3 python/test_analyze.py`
for the self-check of the shot, pacing, caption-grouping and onset logic.

#### `breakdown.json` schema (version 1)

Times are seconds from the start of the video. Fields a backend could not fill are `null`, and a skipped
section has `"status": "skipped"` plus a `"todo"`.

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-09-27T09:00:00",
  "source": {
    "video_path": "/abs/path.mp4", "slug": "big-o-explained-...",
    "duration_s": 59.0, "width": 608, "height": 1080, "fps": 23.976, "has_audio": true,
    "metadata": { /* the download JSON above, or null for local files */ }
  },
  "transcript": {
    "status": "ok|skipped|failed", "backend": "whisper.cpp:ggml-small.en.bin", "language": "en",
    "text": "...",
    "words": [{"text": "If", "start": 0.03, "end": 0.10}],
    "cues": [{"start": 0.03, "end": 2.88, "text": "If you want to get a job ..."}],
    "srt": "transcript.srt"
  },
  "shots": {
    "scene_threshold": 0.3,
    "boundaries": [{"t": 1.585, "type": "cut|dissolve|fade_black|cut_to_black|flash_white", "confidence": 0.9, "score": 0.75}],
    "list": [{
      "index": 1, "start": 0.0, "end": 1.585, "duration": 1.585,
      "transition_in": {"detected": "start|cut|dissolve|fade_black|cut_to_black|flash_white", "confidence": 1.0},
      "spoken": "words whose start falls in this shot",
      "keyframes": {"first": "keyframes/shot001_first.jpg", "mid": "...", "last": "..."},
      "strip": "strips/shot001.jpg",
      "vision": {                                  // null when vision was skipped
        "shot": 1,
        "framing": "talking_head|b_roll|screen_recording|graphic|text_card|meme_or_stock|product|mixed",
        "shot_size": "extreme_close|close|medium|wide|n/a",
        "composition": "...",
        "camera": "static|pan|tilt|zoom_in|zoom_out|punch_in|handheld|tracking|n/a",
        "on_screen_text": ["..."],
        "captions": {"present": true, "position": "top|upper_third|center|lower_third|bottom|none",
                     "words_per_line": 3, "lines": 1, "font_style": "...", "text_color": "...",
                     "highlight_color": "... or null", "animation_guess": "..."},
        "effects": ["..."],
        "transition_in": {"type": "cut|dissolve|fade|whip_pan|zoom|flash|glitch|slide|match_cut|start", "evidence": "..."},
        "description": "..."
      }
    }]
  },
  "pacing": {
    "shot_count": 27, "cut_count": 26, "avg_shot_s": 2.18, "median_shot_s": 1.8,
    "min_shot_s": 0.25, "max_shot_s": 4.5, "cuts_per_10s": 4.4, "cuts_in_first_3s": 2,
    "longest_shot": {"index": 15, "start": 27.99, "duration": 4.5}
  },
  "contact_sheet": "contact_sheet.jpg",
  "audio": {
    "status": "ok|no_audio",
    "loudness": {"integrated_lufs": -19.8, "lra_lu": 2.2, "true_peak_dbfs": -4.3},
    "music": {"music_bed_likely": true, "floor_p10_db": -37.4, "median_db": -23.9, "method": "..."},
    "hits": [{"t": 2.99, "band": "full|high|low", "likely": "hit|whoosh_or_click|impact",
              "rise_db": 18.5, "level_db": -20.1, "nearest_cut_s": 0.15, "aligned_to_cut": true}],
    "sfx_candidate_count": 40, "sfx_aligned_to_cuts": 12, "cuts_with_sfx_ratio": 0.46
  },
  "vision": {"backend": "api|ollama|claude-cli|none", "status": "ok|skipped|failed", "todo": "only when skipped"},
  "captions": {
    "status": "ok|skipped", "source": "vision", "present_ratio": 0.8, "position": "center",
    "words_per_line_median": 3, "lines": 1, "font_style": "...", "text_color": "...",
    "highlight_colors": ["..."], "animation": "..."
  },
  "summary": {
    "hook": {"first_3s_words": "...", "first_3s_shots": 3, "first_3s_on_screen_text": ["..."]},
    "beats": [{"start": 0.03, "end": 2.88, "label": "hook", "summary": "..."}],
    "transition_mix": {"cut": 24, "dissolve": 1},
    "what_to_steal": ["Cut every 2.2s on average ..."],
    "llm": {                                       // only with --vision api or claude-cli
      "hook": {"first_3s_summary": "...", "hook_type": "..."},
      "beats": [{"start": 0.0, "end": 3.0, "label": "hook", "summary": "..."}],
      "format_formula": "...",
      "what_to_steal": ["..."]
    }
  }
}
```

Heuristics worth knowing: SFX hits are level jumps, so speech plosives and sibilants can show up too
(hits with `aligned_to_cut: true` are the reliable ones). The music-bed flag compares the noise floor to
the median level, so a noisy room can read as music. Detected transitions come from pixels; the vision
pass gives a second opinion in `vision.transition_in`.

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

# Crop to vertical (9:16), auto-centers the crop region
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

### Pipeline: AI-powered highlight reels

The `pipeline` command is the full end-to-end workflow. Give it a YouTube URL and a preset, and it will:

1. Download the video
2. Transcribe + diarize (in parallel)
3. **AI highlight selection**, sends transcript to GPT-4o-mini to pick 3-5 compelling segments (60-90s total)
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

Simpler pipeline, downloads the video, cuts a clip, crops to the platform's aspect ratio, scales to target resolution, and optionally burns subtitles:

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
| linkedin | 1080x1350 | 4:5 | 30 | - |
| twitter | 1280x720 | 16:9 | 30 | 140s |
| square | 1080x1080 | 1:1 | 30 | - |

You can add custom presets by creating a new TOML file in `presets/`.

## Architecture

```
raijincut/
├── src/main.rs              # Rust CLI, argument parsing, process orchestration,
│                             #   threading, and all video processing commands
├── python/
│   ├── analyze.py           # `analyze`: transcript, shots, keyframes, vision, audio, breakdown
│   ├── test_analyze.py      # self-check for analyze.py
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
├── Cargo.toml               # Rust dependencies (clap, serde, toml, tempfile)
└── CLAUDE.md                # AI assistant instructions
```

### How it works

The Rust binary is the single entry point. It uses [clap](https://docs.rs/clap) for argument parsing and dispatches to handler functions that:

1. **For simple operations** (cut, crop, convert, probe, download), directly invoke `ffmpeg`/`ffprobe`/`yt-dlp` via `std::process::Command`
2. **For ML operations** (transcribe, rough-cut, subtitle), shell out to Python scripts or embed Python code inline
3. **For pipelines** (pipeline, repurpose), chain multiple steps sequentially, with the `pipeline` command adding AI highlight selection and multi-clip stitching with transitions
4. **For batch work** (batch-transcribe), spawn OS threads that each run independent processing pipelines

The binary finds `python/` and `presets/` in the current directory, next to the executable, or in the source checkout it was built from, so it works from any directory.

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
