#!/usr/bin/env python3
"""
Standalone transcription script — outputs JSON segments.
Called by Rust orchestrator. Takes a local audio file, not a URL.

Usage: python3 transcribe_only.py <audio_file> -o <output.json> [--chunk-duration 1800]
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_CHUNK_DURATION = 600  # 10 minutes (keeps chunks under 25MB Whisper API limit)

# Common Whisper hallucination phrases (appears when model hears music/silence)
HALLUCINATION_PATTERNS = [
    "подпишись",  # Russian "subscribe"
    "subscribe to",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "字幕",  # Chinese "subtitles"
    "翻訳",  # Japanese "translation"
    "ご視聴",  # Japanese "viewing"
    "視聴",
    "구독",  # Korean "subscribe"
    "감사합니다",  # Korean "thank you"
    "sous-titres",  # French "subtitles"
    "untertitel",  # German "subtitles"
    "amara.org",
    "www.",
]


def filter_hallucinations(segments):
    """Remove Whisper hallucination segments (repetitive nonsense during music/silence)."""
    if not segments:
        return segments

    filtered = []
    text_counts = {}

    # Count occurrences of each text
    for seg in segments:
        text = seg.get("text", "").strip().lower()
        text_counts[text] = text_counts.get(text, 0) + 1

    # Determine dominant language by sampling middle segments (skip intro/outro)
    mid = len(segments) // 2
    sample = segments[max(0, mid - 20):mid + 20]

    removed = 0
    for seg in segments:
        text = seg.get("text", "").strip()
        text_lower = text.lower()

        # Skip empty segments
        if not text:
            removed += 1
            continue

        # Filter: exact text repeated 3+ times across the transcript
        if text_counts.get(text_lower, 0) >= 3:
            removed += 1
            continue

        # Filter: known hallucination phrases
        if any(pattern in text_lower for pattern in HALLUCINATION_PATTERNS):
            removed += 1
            continue

        filtered.append(seg)

    if removed > 0:
        print(f"🧹 Filtered {removed} hallucinated segments")

    return filtered


def get_api_key():
    config_path = os.path.expanduser("~/.openclaw/video-transcribe-config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            return config.get("openai_api_key")
    return os.environ.get("OPENAI_API_KEY")


def get_audio_duration(audio_file):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio(audio_file, chunk_duration, output_dir):
    total_duration = get_audio_duration(audio_file)
    num_chunks = max(1, int((total_duration + chunk_duration - 1) // chunk_duration))

    if num_chunks == 1:
        return [audio_file]

    print(f"✂️ Splitting into {num_chunks} chunks...")
    chunk_files = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_file = os.path.join(output_dir, f"chunk_{i+1:03d}.mp3")
        cmd = [
            "ffmpeg", "-y", "-i", audio_file,
            "-ss", str(start_time), "-t", str(chunk_duration),
            "-acodec", "copy", chunk_file
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunk_files.append(chunk_file)
        print(f"   ✅ Chunk {i+1}/{num_chunks}")

    return chunk_files


def transcribe_file(audio_file, api_key):
    """Transcribe a single audio file, return segments list."""
    import requests

    duration_min = get_audio_duration(audio_file) / 60
    cost = duration_min * 0.006
    print(f"⏱️ Duration: {duration_min:.1f} min (est. cost: ${cost:.4f})")

    print("🔄 Transcribing with OpenAI Whisper API...")
    with open(audio_file, "rb") as f:
        files = {
            "file": (os.path.basename(audio_file), f, "audio/mpeg"),
            "model": (None, "whisper-1"),
            "response_format": (None, "verbose_json"),
            "timestamp_granularities": (None, "segment")
        }
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files, timeout=600
        )

    if response.status_code != 200:
        print(f"❌ API Error ({response.status_code}): {response.text[:500]}", file=sys.stderr)
        sys.exit(1)

    result = response.json()
    return result.get("segments", []), cost


def main():
    parser = argparse.ArgumentParser(description="Standalone transcription to JSON")
    parser.add_argument("audio_file", help="Path to local audio file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("--chunk-duration", type=int, default=DEFAULT_CHUNK_DURATION,
                        help="Chunk duration in seconds for long audio")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY or configure ~/.openclaw/video-transcribe-config.json")
        sys.exit(1)

    total_duration = get_audio_duration(args.audio_file)
    num_chunks = max(1, int((total_duration + args.chunk_duration - 1) // args.chunk_duration))

    all_segments = []
    total_cost = 0

    if num_chunks > 1:
        chunk_dir = os.path.dirname(args.audio_file) or "/tmp"
        chunk_files = split_audio(args.audio_file, args.chunk_duration, chunk_dir)

        for i, chunk_file in enumerate(chunk_files):
            print(f"   📝 Chunk {i+1}/{len(chunk_files)}...")
            chunk_start = i * args.chunk_duration
            segments, cost = transcribe_file(chunk_file, api_key)
            total_cost += cost
            for seg in segments:
                seg["start"] += chunk_start
                seg["end"] += chunk_start
                all_segments.append(seg)
    else:
        segments, cost = transcribe_file(args.audio_file, api_key)
        total_cost = cost
        all_segments = segments

    # Filter Whisper hallucinations before writing
    all_segments = filter_hallucinations(all_segments)

    # Write output JSON
    output = {
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg.get("text", "")}
            for seg in all_segments
        ]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Transcription saved to {args.output} (cost: ${total_cost:.4f})")


if __name__ == "__main__":
    main()
