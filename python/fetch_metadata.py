#!/usr/bin/env python3
"""
Standalone metadata extraction script — outputs JSON.
Extracts video metadata via yt-dlp without downloading.

Usage: python3 fetch_metadata.py <url> -o <metadata.json>
"""

import argparse
import json
import subprocess
import sys


def get_video_metadata(url):
    cmd = [
        "yt-dlp",
        "--print", "%(title)s",
        "--print", "%(duration)s",
        "--print", "%(uploader)s",
        "--print", "%(upload_date)s",
        "--print", "%(view_count)s",
        "--print", "%(like_count)s",
        "--print", "%(channel_id)s",
        "--print", "%(playlist_title)s",
        "--print", "%(id)s",
        "--no-download",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]

    metadata = {
        "title": lines[0] if len(lines) > 0 else "Unknown",
        "duration": int(lines[1]) if len(lines) > 1 else 0,
        "uploader": lines[2] if len(lines) > 2 else "Unknown",
        "upload_date": lines[3] if len(lines) > 3 else "",
        "view_count": lines[4] if len(lines) > 4 else "0",
        "like_count": lines[5] if len(lines) > 5 else "0",
        "channel_id": lines[6] if len(lines) > 6 else "",
        "playlist_title": lines[7] if len(lines) > 7 else "",
        "video_id": lines[8] if len(lines) > 8 else "",
        "url": url
    }

    # Format duration
    duration_sec = metadata["duration"]
    metadata["duration_formatted"] = f"{duration_sec//60} min {duration_sec%60} sec"

    # Format date
    date_str = metadata["upload_date"]
    if len(date_str) == 8:
        try:
            metadata["upload_date_formatted"] = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
        except Exception:
            metadata["upload_date_formatted"] = date_str
    else:
        metadata["upload_date_formatted"] = date_str

    # Format counts
    try:
        metadata["view_count_formatted"] = f"{int(metadata['view_count']):,}"
    except (ValueError, TypeError):
        metadata["view_count_formatted"] = metadata["view_count"]

    try:
        metadata["like_count_formatted"] = f"{int(metadata['like_count']):,}"
    except (ValueError, TypeError):
        metadata["like_count_formatted"] = metadata["like_count"]

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fetch video metadata as JSON")
    parser.add_argument("url", help="Video URL")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"🔍 Fetching metadata for {args.url}...")
    metadata = get_video_metadata(args.url)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"📹 {metadata['title']} ({metadata['duration_formatted']})")
    print(f"✅ Metadata saved to {args.output}")


if __name__ == "__main__":
    main()
