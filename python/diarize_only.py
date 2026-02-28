#!/usr/bin/env python3
"""
Standalone diarization script — outputs JSON speaker segments.
Called by Rust orchestrator. Uses speaker_detector.py (ffmpeg-based, no torchaudio).

Usage: python3 diarize_only.py <audio_file> -o <output.json>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Ensure sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speaker_detector import detect_speakers, merge_adjacent_segments


def convert_to_wav(audio_file, output_dir):
    """Convert any audio format to 16kHz mono WAV for pyannote compatibility."""
    wav_path = os.path.join(output_dir, "diarize_input.wav")
    cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert audio to WAV: {result.stderr}")
    return wav_path


def main():
    parser = argparse.ArgumentParser(description="Standalone diarization to JSON")
    parser.add_argument("audio_file", help="Path to local audio file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Always convert to 16kHz mono WAV for pyannote (avoids sample count mismatches)
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"🎤 Preparing audio for diarization...")
        wav_file = convert_to_wav(args.audio_file, tmp_dir)
        print(f"✅ Audio converted to 16kHz WAV")

        print(f"🔍 Running speaker diarization...")
        segments = detect_speakers(
            wav_file,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        merged = merge_adjacent_segments(segments)

    output = {
        "speakers": [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker}
            for seg in merged
        ]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Clean up GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"✅ Diarization saved to {args.output} ({len(merged)} segments)")


if __name__ == "__main__":
    main()
