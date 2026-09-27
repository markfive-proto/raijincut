# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build --release        # release binary at target/release/raijincut
cargo run -- <subcommand>    # run in dev mode
cargo run -- --help          # show CLI usage
```

Self-check for the analyze logic: `python3 python/test_analyze.py`. No linter configuration is set up.

## Architecture

**Rust CLI (`src/main.rs`)**: Single-file CLI using clap derive macros. Subcommands dispatch to standalone functions that shell out to external tools (`ffmpeg`, `ffprobe`, `yt-dlp`) or invoke Python scripts inline via `Command::new("python3").args(["-c", ...])`. There is no library crate; everything is in `main.rs`.

**Python scripts (`python/`)**: Standalone scripts for advanced features: transcription (Whisper/OpenAI), repurposing pipelines, filler word removal, speaker diarization, audio analysis, and transitions. The Rust CLI references some of these but mostly embeds Python code as inline strings rather than calling the script files directly.

**Presets (`presets/*.toml`)**: Platform-specific video settings (TikTok, Reels, Shorts, LinkedIn, Twitter, square). Each preset defines `[video]` (resolution, aspect ratio, codec, bitrate, fps) and `[subtitle]` (font, color, position) sections. The preset system is referenced by the `repurpose` command but not yet fully wired up in Rust.

## External Dependencies

Runtime requires: `ffmpeg`, `ffprobe`, `yt-dlp`, `python3` with `moviepy` and `pysubs2` packages.

`analyze` (`python/analyze.py`) is stdlib-only Python (Pillow optional) and shells out to ffmpeg, whisper-cli or mlx_whisper, and a vision backend (Claude API via `ANTHROPIC_API_KEY`, ollama, or the `claude` CLI). Its `breakdown.json` schema is documented in README.md; bump `SCHEMA_VERSION` when it changes. `download` writes `<video>.json` metadata next to the mp4, which `analyze` picks up automatically.

## Key Patterns

- All video processing functions follow the same pattern: print status, spawn external process via `std::process::Command`, print result. Error handling uses `.unwrap()` throughout.
- The `subtitle` and `rough-cut` commands embed multi-line Python scripts as format strings, substituting file paths directly into the Python source.
- Aspect ratio cropping computes crop dimensions from ffprobe output, centering the crop region.
