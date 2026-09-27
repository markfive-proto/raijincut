# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install, run, test

```bash
uv tool install -e .              # or pipx install -e . ; puts `raijincut` on PATH (editable)
python3 -m raijincut --help       # run from the checkout without installing
python3 tests/test_raijincut.py   # one self-check per command on ffmpeg lavfi samples
```

No linter configuration is set up.

## Architecture

Pure Python package `raijincut/`, entry point `raijincut.cli:main` (argparse).

- `cli.py`: download (yt-dlp), probe, cut, crop, convert, subtitle burn, repurpose, presets. Each edit is one
  direct ffmpeg call. Subtitle burn draws cue PNGs with Pillow and overlays them via the concat demuxer, so it
  works on ffmpeg builds without libass/drawtext.
- `analyze.py`: the `analyze` breakdown (transcript, shots, keyframes, vision, audio, summary). Also holds the
  shared helpers `transcribe_video`, `group_words`, `write_srt`, `parse_srt`.
- `roughcut.py`: `rough-cut`, transcript-driven removal of fillers, repeats and long pauses. It is the single
  implementation; solopreneur video-studio's `cleanup` tool shells out to it.
- `presets/*.toml`: platform presets (`[video]`, `[subtitle]`), shipped as package data.

## Rules

- No moviepy. Edits are direct ffmpeg calls.
- No PyTorch / openai-whisper. Transcripts: whisper.cpp (`--whisper-model` / `RAIJINCUT_WHISPER_MODEL`) or mlx_whisper.
- Never load full-resolution frames into Python. Stream downscaled frames (<= 160 px) and audio from ffmpeg pipes
  one chunk at a time (`analyze.stream`).
- `breakdown.json` schema is documented in README.md; bump `SCHEMA_VERSION` when it changes. `download` writes
  `<video>.json` metadata next to the mp4, which `analyze` picks up automatically.
