#!/usr/bin/env python3
"""
Video Transcriber CLI - Enhanced with Long Video Support & Full Metadata & Speaker Diarization
- Auto-fetches YouTube metadata (title, channel, date, views, etc.)
- Uses original video title as filename
- Rich metadata header in transcription
- Splits long videos into chunks before transcription
- Speaker diarization using pyannote.audio (optional, use --diarize flag)
Usage: python video_transcribe.py <url> [--output /path/to/output] [--diarize]
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
import re
from pathlib import Path
from datetime import datetime

# Default output path - 2nd Brain
DEFAULT_OUTPUT = os.path.expanduser("~/Documents/2ndbrain/2ndBrain/Tasks/Videos/Transcripts")

# Chunk settings (in seconds)
DEFAULT_CHUNK_DURATION = 600  # 10 minutes (keeps chunks under 25MB Whisper API limit)



def sanitize_filename(title: str, max_length: int = 60) -> str:
    """Convert title to safe filename"""
    # Replace invalid chars with underscore
    safe = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    # Truncate if too long
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_')
    return safe

def get_video_metadata(url: str) -> dict:
    """Get full video metadata from YouTube"""
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
    
    # Format date (YYYYMMDD -> readable)
    date_str = metadata["upload_date"]
    if len(date_str) == 8:
        try:
            metadata["upload_date_formatted"] = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
        except:
            metadata["upload_date_formatted"] = date_str
    else:
        metadata["upload_date_formatted"] = date_str
    
    # Format views
    try:
        metadata["view_count_formatted"] = f"{int(metadata['view_count']):,}"
    except:
        metadata["view_count_formatted"] = metadata["view_count"]
    
    try:
        metadata["like_count_formatted"] = f"{int(metadata['like_count']):,}"
    except:
        metadata["like_count_formatted"] = metadata["like_count"]
    
    return metadata

def download_video_audio(url: str, temp_dir: str, filename: str = "audio") -> str:
    """Download video audio using yt-dlp"""
    output_file = os.path.join(temp_dir, f"{filename}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_file,
        url
    ]
    
    print(f"📥 Downloading audio...")
    subprocess.run(cmd, check=True, cwd=temp_dir)
    
    # Find the downloaded file
    for f in os.listdir(temp_dir):
        if f.startswith(f"{filename}."):
            return os.path.join(temp_dir, f)
    
    raise Exception("Failed to download audio")

def split_audio(audio_file: str, chunk_duration: int = DEFAULT_CHUNK_DURATION, output_dir: str = None) -> list:
    """Split audio into chunks"""
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    # Get audio duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    total_duration = float(result.stdout.strip())
    
    print(f"📊 Audio duration: {int(total_duration//60)} min {int(total_duration%60)} sec")
    
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
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-acodec", "copy",
            chunk_file
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunk_files.append(chunk_file)
        print(f"   ✅ Chunk {i+1}/{num_chunks}")
    
    return chunk_files

def transcribe_with_api(audio_file: str, output_file: str, metadata: dict):
    """Transcribe using OpenAI Whisper API"""
    import requests
    
    # Get API key
    config_path = os.path.expanduser("~/.openclaw/video-transcribe-config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            api_key = config.get("openai_api_key")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY or run --set-key")
        sys.exit(1)
    
    # Get duration for cost estimate
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration_min = float(result.stdout.strip()) / 60
    cost = duration_min * 0.006
    
    print(f"⏱️ Duration: {duration_min:.1f} min (est. cost: ${cost:.4f})")
    
    # Transcribe with timestamps
    print("🔄 Transcribing with OpenAI Whisper API (with timestamps)...")
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
        print(f"❌ API Error: {response.text[:200]}")
        sys.exit(1)
    
    result = response.json()
    
    # Build text with timestamps from segments
    segments = result.get("segments", [])
    text_with_timestamps = ""
    if segments:
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            start_formatted = f"{int(start//60):02d}:{int(start%60):02d}"
            end_formatted = f"{int(end//60):02d}:{int(end%60):02d}"
            text_with_timestamps += f"**[{start_formatted} - {end_formatted}]**  \n{text.strip()}\n\n"
    else:
        text_with_timestamps = result.get("text", "")
    
    # Write transcription with full metadata
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"title: \"{metadata['title']}\"\n")
        f.write(f"channel: \"{metadata['uploader']}\"\n")
        f.write(f"video_id: \"{metadata['video_id']}\"\n")
        f.write(f"url: \"{metadata['url']}\"\n")
        f.write(f"date: \"{metadata['upload_date_formatted']}\"\n")
        f.write(f"duration: \"{metadata['duration_formatted']}\"\n")
        f.write(f"views: \"{metadata['view_count_formatted']}\"\n")
        f.write(f"likes: \"{metadata['like_count_formatted']}\"\n")
        f.write(f"transcribed_at: \"{datetime.now().strftime('%Y-%m-%d %H:%M')}\"\n")
        f.write(f"type: transcription\n")
        f.write(f"tags: [video, youtube, transcription]\n")
        f.write("---\n\n")
        
        f.write(f"# 🎬 {metadata['title']}\n\n")
        f.write(f"**Channel:** {metadata['uploader']}\n")
        f.write(f"**Date:** {metadata['upload_date_formatted']}\n")
        f.write(f"**Duration:** {metadata['duration_formatted']}\n")
        f.write(f"**Views:** {metadata['view_count_formatted']}\n")
        f.write(f"**Likes:** {metadata['like_count_formatted']}\n")
        f.write(f"**URL:** {metadata['url']}\n")
        f.write(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"---\n\n")
        f.write(f"## 📝 Transcription (with timestamps)\n\n")
        f.write(f"{text_with_timestamps}\n")
    
    print(f"✅ Done! Cost: ${cost:.4f}")

def transcribe_chunked_with_api(audio_file: str, output_file: str, metadata: dict, chunk_duration: int = DEFAULT_CHUNK_DURATION):
    """Transcribe long audio by splitting into chunks. Takes a local audio path."""
    import requests

    # Get API key
    config_path = os.path.expanduser("~/.openclaw/video-transcribe-config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            api_key = config.get("openai_api_key")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ No API key found")
        sys.exit(1)

    total_duration = metadata["duration"]
    num_chunks = (total_duration + chunk_duration - 1) // chunk_duration

    print(f"🔪 Long video ({metadata['duration_formatted']}) - splitting into {num_chunks} chunks...")

    temp_dir = os.path.dirname(audio_file)

    # Split
    chunk_files = split_audio(audio_file, chunk_duration, temp_dir)

    all_segments = []
    total_cost = 0

    # Transcribe each chunk with timestamps
    for i, chunk_file in enumerate(chunk_files):
        print(f"   📝 Chunk {i+1}/{len(chunk_files)}...")

        # Get chunk start time
        chunk_start = i * chunk_duration

        # Get chunk duration
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", chunk_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        chunk_dur = float(result.stdout.strip()) / 60
        chunk_cost = chunk_dur * 0.006
        total_cost += chunk_cost

        # Transcribe with timestamps
        with open(chunk_file, "rb") as f:
            files = {
                "file": (os.path.basename(chunk_file), f, "audio/mpeg"),
                "model": (None, "whisper-1"),
                "response_format": (None, "verbose_json"),
                "timestamp_granularities": (None, "segment")
            }
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files, timeout=600
            )

        if response.status_code == 200:
            result = response.json()
            # Adjust timestamps by adding chunk start offset
            for seg in result.get("segments", []):
                seg["start"] += chunk_start
                seg["end"] += chunk_start
                all_segments.append(seg)
        else:
            print(f"   ⚠️ Chunk {i+1} failed: {response.text[:100]}")

    # Write merged transcription with timestamps
    text_with_timestamps = ""
    for seg in all_segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        start_formatted = f"{int(start//60):02d}:{int(start%60):02d}"
        end_formatted = f"{int(end//60):02d}:{int(end%60):02d}"
        text_with_timestamps += f"**[{start_formatted} - {end_formatted}]**  \n{text.strip()}\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"title: \"{metadata['title']}\"\n")
        f.write(f"channel: \"{metadata['uploader']}\"\n")
        f.write(f"video_id: \"{metadata['video_id']}\"\n")
        f.write(f"url: \"{metadata['url']}\"\n")
        f.write(f"date: \"{metadata['upload_date_formatted']}\"\n")
        f.write(f"duration: \"{metadata['duration_formatted']}\"\n")
        f.write(f"views: \"{metadata['view_count_formatted']}\"\n")
        f.write(f"likes: \"{metadata['like_count_formatted']}\"\n")
        f.write(f"chunks: {len(chunk_files)}\n")
        f.write(f"transcribed_at: \"{datetime.now().strftime('%Y-%m-%d %H:%M')}\"\n")
        f.write(f"type: transcription\n")
        f.write(f"tags: [video, youtube, transcription]\n")
        f.write("---\n\n")

        f.write(f"# 🎬 {metadata['title']}\n\n")
        f.write(f"**Channel:** {metadata['uploader']}\n")
        f.write(f"**Date:** {metadata['upload_date_formatted']}\n")
        f.write(f"**Duration:** {metadata['duration_formatted']}\n")
        f.write(f"**Views:** {metadata['view_count_formatted']}\n")
        f.write(f"**Likes:** {metadata['like_count_formatted']}\n")
        f.write(f"**URL:** {metadata['url']}\n")
        f.write(f"**Chunks:** {len(chunk_files)}\n")
        f.write(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"---\n\n")
        f.write(f"## 📝 Transcription (with timestamps)\n\n")
        f.write(f"{text_with_timestamps}\n")

    print(f"✅ Done! {len(chunk_files)} chunks, total cost: ${total_cost:.4f}")

def _add_speakers_to_transcript(transcript_file: str, speakers: dict):
    """Add speaker labels to transcript based on diarization results."""
    if not speakers:
        return

    with open(transcript_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'\*\*\[(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})\]\*\*\s*\n(.+?)(?=\n\n|\Z)'

    def get_time_seconds(m, s):
        return int(m) * 60 + int(s)

    def find_speaker(time_sec, speakers):
        for speaker, segs in speakers.items():
            for start, end in segs:
                if start <= time_sec <= end:
                    return speaker
        return None

    def replace_with_speaker(match):
        start_min, start_sec, end_min, end_sec = match.groups()[:4]
        text = match.group(5).strip()
        time_start = get_time_seconds(start_min, start_sec)

        speaker = find_speaker(time_start, speakers)
        if speaker:
            short_speaker = f"Speaker {chr(65 + int(speaker.split('_')[-1]))}"
            return f"**[{start_min}:{start_sec} - {end_min}:{end_sec}]** 👤 {short_speaker}\n{text}"
        else:
            return match.group(0)

    new_content = re.sub(pattern, replace_with_speaker, content, flags=re.DOTALL)

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"   📝 Speaker labels added to transcript")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos with full metadata"
    )
    parser.add_argument("url", help="Video URL to transcribe")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output directory")
    parser.add_argument("--chunk-duration", "-c", type=int, default=DEFAULT_CHUNK_DURATION, help="Chunk duration in seconds")
    parser.add_argument("--set-key", metavar="KEY", help="Save OpenAI API key")
    parser.add_argument("--diarize", "-d", action="store_true", help="Enable speaker diarization (requires HuggingFace token)")
    
    args = parser.parse_args()
    
    # Handle API key set
    if args.set_key:
        config_path = os.path.expanduser("~/.openclaw/video-transcribe-config.json")
        with open(config_path, "w") as f:
            json.dump({"openai_api_key": args.set_key}, f)
        print(f"✅ API key saved to {config_path}")
        return
    
    # Get full metadata
    print(f"🔍 Fetching video metadata...")
    try:
        metadata = get_video_metadata(args.url)
        print(f"📹 Title: {metadata['title']}")
        print(f"👤 Channel: {metadata['uploader']}")
        print(f"📅 Date: {metadata['upload_date_formatted']}")
        print(f"⏱️  Duration: {metadata['duration_formatted']}")
        print(f"👁️  Views: {metadata['view_count_formatted']}")
    except Exception as e:
        print(f"❌ Could not fetch metadata: {e}")
        return
    
    # Generate filename from title
    safe_filename = sanitize_filename(metadata["title"])
    output_file = os.path.join(args.output, f"transcript_{safe_filename}.md")
    
    # Handle duplicate filenames
    counter = 1
    while os.path.exists(output_file):
        output_file = os.path.join(args.output, f"transcript_{safe_filename}_{counter}.md")
        counter += 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Transcribe (single download, reused for diarization if needed)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio once
            audio_file = download_video_audio(args.url, temp_dir)

            # Transcribe
            if metadata["duration"] > args.chunk_duration:
                transcribe_chunked_with_api(audio_file, output_file, metadata, args.chunk_duration)
            else:
                transcribe_with_api(audio_file, output_file, metadata)

            # Run speaker diarization if requested (reuses same audio file)
            if args.diarize:
                print(f"🎤 Running speaker diarization...")
                # Import from speaker_detector (uses ffmpeg, not torchaudio)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from speaker_detector import detect_speakers, segments_to_speaker_dict, merge_adjacent_segments

                segments = detect_speakers(audio_file)
                merged = merge_adjacent_segments(segments)
                speakers = segments_to_speaker_dict(merged)

                if speakers:
                    print(f"   👥 Detected {len(speakers)} speakers")
                    # Inline speaker label insertion
                    _add_speakers_to_transcript(output_file, speakers)

        print(f"\n🎉 Saved to:")
        print(f"   {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
