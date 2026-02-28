#!/usr/bin/env python3
"""
Merge transcription JSON + optional diarization JSON + metadata JSON → Markdown output.
Called by Rust orchestrator as the final pipeline step.

Usage: python3 merge_transcript.py <transcript.json> -o <output.md> [-d <diarization.json>] [-m <metadata.json>]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_speaker(time_sec, speakers_dict):
    """Find which speaker is talking at a given timestamp."""
    for speaker, segments in speakers_dict.items():
        for start, end in segments:
            if start <= time_sec <= end:
                return speaker
    return None


def format_speaker(speaker_label):
    """Convert SPEAKER_00 → Speaker A."""
    try:
        idx = int(speaker_label.split("_")[-1])
        return f"Speaker {chr(65 + idx)}"
    except (ValueError, IndexError):
        return speaker_label


def main():
    parser = argparse.ArgumentParser(description="Merge transcription + diarization → Markdown")
    parser.add_argument("transcript", help="Transcription JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output Markdown file")
    parser.add_argument("-d", "--diarization", help="Diarization JSON file (optional)")
    parser.add_argument("-m", "--metadata", help="Metadata JSON file (optional)")
    args = parser.parse_args()

    # Load transcription
    transcript = load_json(args.transcript)
    segments = transcript.get("segments", [])

    # Load diarization if provided
    speakers_dict = {}
    if args.diarization and os.path.exists(args.diarization):
        diarization = load_json(args.diarization)
        for entry in diarization.get("speakers", []):
            label = entry["speaker"]
            if label not in speakers_dict:
                speakers_dict[label] = []
            speakers_dict[label].append((entry["start"], entry["end"]))

    # Load metadata if provided
    metadata = {}
    if args.metadata and os.path.exists(args.metadata):
        metadata = load_json(args.metadata)

    # Build markdown
    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f'title: "{metadata.get("title", "Untitled")}"')
    lines.append(f'channel: "{metadata.get("uploader", "Unknown")}"')
    lines.append(f'video_id: "{metadata.get("video_id", "")}"')
    lines.append(f'url: "{metadata.get("url", "")}"')
    lines.append(f'date: "{metadata.get("upload_date_formatted", "")}"')
    lines.append(f'duration: "{metadata.get("duration_formatted", "")}"')
    lines.append(f'views: "{metadata.get("view_count_formatted", "")}"')
    lines.append(f'likes: "{metadata.get("like_count_formatted", "")}"')
    lines.append(f'transcribed_at: "{datetime.now().strftime("%Y-%m-%d %H:%M")}"')
    if speakers_dict:
        lines.append(f"speakers: {len(speakers_dict)}")
    lines.append("type: transcription")
    lines.append("tags: [video, youtube, transcription]")
    lines.append("---\n")

    # Header
    title = metadata.get("title", "Untitled")
    lines.append(f"# 🎬 {title}\n")
    if metadata:
        lines.append(f"**Channel:** {metadata.get('uploader', 'Unknown')}")
        lines.append(f"**Date:** {metadata.get('upload_date_formatted', '')}")
        lines.append(f"**Duration:** {metadata.get('duration_formatted', '')}")
        lines.append(f"**Views:** {metadata.get('view_count_formatted', '')}")
        lines.append(f"**Likes:** {metadata.get('like_count_formatted', '')}")
        lines.append(f"**URL:** {metadata.get('url', '')}")
        lines.append(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")
    lines.append("## 📝 Transcription (with timestamps)\n")

    # Filter any remaining hallucinations (repeated identical text)
    text_counts = {}
    for seg in segments:
        t = seg.get("text", "").strip().lower()
        text_counts[t] = text_counts.get(t, 0) + 1
    segments = [s for s in segments if s.get("text", "").strip() and text_counts.get(s["text"].strip().lower(), 0) < 3]

    # Segments
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()

        start_fmt = f"{int(start//60):02d}:{int(start%60):02d}"
        end_fmt = f"{int(end//60):02d}:{int(end%60):02d}"

        if speakers_dict:
            speaker = find_speaker(start, speakers_dict)
            if speaker:
                short = format_speaker(speaker)
                lines.append(f"**[{start_fmt} - {end_fmt}]** 👤 {short}  ")
            else:
                lines.append(f"**[{start_fmt} - {end_fmt}]**  ")
        else:
            lines.append(f"**[{start_fmt} - {end_fmt}]**  ")

        lines.append(f"{text}\n")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Merged transcript saved to {args.output}")
    if speakers_dict:
        print(f"   👥 {len(speakers_dict)} speakers labeled")


if __name__ == "__main__":
    main()
