#!/usr/bin/env python3
"""
Video Repurpose CLI - Trim, clip and repurpose videos using MoviePy

Usage:
    video-repurpose "URL" --clips "0:00-2:30,5:00-7:30"     # Extract clips
    video-repurpose "URL" --auto                              # Auto-detect highlights
    video-repurpose "URL" --rough-cut                         # Remove silence/fillers
    video-repurpose "URL" --subtitles                        # Burn in subtitles
    video-repurpose "URL" --transition fade                  # Add transitions
    video-repurpose "URL" --reel                             # Create reel
"""

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

# MoviePy for video editing
from moviepy import VideoFileClip, concatenate_videoclips, AudioClip, TextClip, CompositeVideoClip, ImageClip
import moviepy.video.fx as vfx

DEFAULT_QUALITY = "720p"
DEFAULT_TRANSCRIPT_DIR = os.path.expanduser("~/Documents/2ndbrain/2ndBrain/Tasks/Videos/Transcripts")
DEFAULT_OUTPUT = os.path.expanduser("~/Documents/2ndbrain/2ndBrain/Tasks/Videos/Clips")

# Filler words to remove
FILLER_PATTERNS = [
    r'\b(?:um|uh|er|ah|like you know)\b',
    r'\b(you know)\s*(?:\1){1,3}',  # repeated "you know"
    r'\s{2,}',  # multiple spaces
]

@dataclass
class Segment:
    """Transcript segment with timestamp"""
    start: float
    end: float
    text: str
    score: float = 0.0
    reason: str = ""

# Viral/Engagement keywords for scoring - ENHANCED
VIRAL_KEYWORDS = {
    # Controversy/Attention grabbers
    "controversy": 2.5, "disagree": 2.2, "uncomfortable": 2.0,
    "most relevant": 2.8, "important": 1.8, "critical": 1.8,
    "dangerous": 2.5, "risk": 1.8, "fear": 1.8, "worried": 1.8,
    
    # Predictions/Future
    "will": 1.5, "going to": 1.5, "predict": 2.2, "future": 2.0,
    "expect": 1.8, "probably": 1.3, "maybe": 1.2,
    
    # Advice/Actionable
    "advice": 3.0, "should": 2.0, "recommend": 2.5, 
    "tip": 2.5, "learn": 1.8, "how to": 2.0, "recommendation": 2.5,
    "my advice": 3.0, "i recommend": 3.0, "i would say": 2.0,
    
    # Numbers/Impact
    "billion": 2.8, "million": 2.5, "percent": 2.5,
    "10x": 3.0, "100x": 3.0, "10,000": 2.2,
    
    # AI specific
    "AI": 1.5, "artificial intelligence": 1.8, "model": 1.5,
    "AGI": 2.5, "artificial general intelligence": 2.8,
    
    # Career/Jobs
    "career": 2.5, "job": 2.2, "jobs": 2.2, "employment": 2.2,
    "replace": 2.0, "disrupt": 2.2, "disruption": 2.2,
    "opportunity": 2.0, "startup": 1.8, "entrepreneur": 1.8,
    
    # Country/Economy
    "india": 2.0, "indian": 2.0, "country": 1.8, "economy": 2.2, "economic": 2.0,
    
    # Science/Tech
    "biotech": 2.8, "biology": 2.5, "health": 2.0, "medicine": 2.2,
    "cancer": 2.8, "cure": 2.8, "disease": 2.5,
    
    # Regulation/Policy
    "regulation": 2.2, "government": 1.8, "policy": 1.8, "law": 1.8,
    
    # Companies
    "open source": 2.5, "google": 1.5, "openai": 1.5, "anthropic": 1.5,
    "deepseek": 2.0, "china": 1.8, "american": 1.5,
    
    # Consciousness/AI identity
    "consciousness": 2.5, "sentient": 2.2, "think": 1.5, "believe": 1.3,
    
    # Power/Control
    "power": 2.0, "control": 1.8, "concentration": 2.0, "moat": 2.8,
    "competitive": 2.0, "advantage": 2.0,
    
    # Society/World
    "human": 1.5, "society": 2.0, "world": 1.5, "people": 1.3,
    "stupider": 2.5, "dumber": 2.2,  # Negative outcomes
}

# ANSWER indicators for interview/podcast detection
ANSWER_INDICATORS = [
    r"^I think", r"^I believe", r"^I would say", r"^I mean",
    r"^My view", r"^I don't think", r"^I agree", r"^I disagree",
    r"^The key", r"^One is", r"^I want to", r"^Let me",
    r"^Here's", r"^So,", r"^You know,",
]

# QUESTION patterns to deprioritize
QUESTION_PATTERNS = [
    r"^\w+\s+do\s+you", r"^\w+\s+what\s+is", r"^\w+\s+how\s+do",
    r"^\w+\s+can\s+you", r"^\w+\s+will\s+you", r"^\w+\s+would\s+you",
    r"^\w+\s+do\s+you\s+think", r"^\w+\s+are\s+you",
]

SKIP_PATTERNS = [
    r"ПОДПИШИСЬ",
    r"subscribe",
    r"like and subscribe",
    r"please like",
    r"don't forget",
    r"hit the",
    r"bell icon",
    r"thank you for watching",
]

TRANSITIONS = {
    "fade": "Crossfade (0.5s black)",
    "wipe-left": "Wipe from right (0.5s)",
    "wipe-right": "Wipe from left (0.5s)",
    "slide-left": "Slide left (0.5s)",
    "slide-right": "Slide right (0.5s)",
    "zoom": "Quick zoom in/out (0.3s)",
    "dissolve": "Soft crossfade (0.7s)",
    "blur": "Blur transition (0.5s)",
    "none": "No transition (cut)",
}

def get_video_info(url: str) -> dict:
    """Get video info using yt-dlp"""
    cmd = [
        "yt-dlp", "--dump-json", "--no-download", 
        "--cookies-from-browser", "brave",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "duration": data.get("duration", 0),
        "uploader": data.get("uploader"),
        "url": url
    }

def parse_timestamp(ts: str) -> float:
    """Parse timestamp string to seconds (supports MM:SS and HH:MM:SS formats)"""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(ts)

def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp format"""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def find_transcript(video_id: str) -> str:
    """Find transcript file for video ID"""
    transcript_dir = Path(DEFAULT_TRANSCRIPT_DIR)
    if not transcript_dir.exists():
        return ""
    
    for f in transcript_dir.glob(f"*{video_id}*"):
        if f.suffix == ".md":
            return str(f)
    
    for f in transcript_dir.glob("*.md"):
        if video_id in f.stem:
            return str(f)
    
    return ""

def parse_transcript(transcript_path: str) -> List[Segment]:
    """Parse transcript file and extract segments with timestamps"""
    segments = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: **[MM:SS - MM:SS]** followed by text
    # Note: No colon after the first **[ - actual format is **[00:00 - 00:07]**
    pattern = r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*\s*\n(.*?)(?=\n\*\*\[\d|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for start_ts, end_ts, text in matches:
        text = text.strip()
        
        should_skip = False
        for skip in SKIP_PATTERNS:
            if re.search(skip, text, re.IGNORECASE):
                should_skip = True
                break
        
        if should_skip or len(text) < 10:
            continue
        
        start = parse_timestamp(start_ts)
        end = parse_timestamp(end_ts)
        
        if end - start < 2 or end - start > 300:
            continue
            
        segments.append(Segment(start=start, end=end, text=text))
    
    # If still empty, try alternative pattern without **
    if not segments:
        pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*\n(.*?)(?=\n\[\d|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for start_ts, end_ts, text in matches:
            text = text.strip()
            if len(text) < 10:
                continue
            start = parse_timestamp(start_ts)
            end = parse_timestamp(end_ts)
            if end > start:
                segments.append(Segment(start=start, end=end, text=text))
    
    return segments

# ========== ROUGH CUT FUNCTIONS ==========

def remove_filler_words(text: str) -> str:
    """Remove filler words from text: um, uh, er, ah, like, you know"""
    # Remove individual fillers with word boundaries
    text = re.sub(r'\b(?:um|uh|er|ah|hm|mmm)\b', '', text, flags=re.IGNORECASE)
    
    # Remove repeated "you know" (keep first only)
    text = re.sub(r'(\byou know\b)(?:\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    
    # Remove "like" when used as filler (not as comparison)
    # This is tricky - be conservative and only remove clear fillers
    text = re.sub(r'\blike,\s+', 'like ', text)  # "like," followed by space
    text = re.sub(r'\s+like\s+like\s+', ' like ', text)  # repeated like
    
    # Remove repeated single words (3+ times)
    text = re.sub(r'\b(\w+)(?:\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation artifacts
    text = re.sub(r'\s*,\s*,', ',', text)  # double commas
    text = re.sub(r'\s*,\s*\.', '.', text)  # comma before period
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    return text

def remove_repeated_phrases(text: str) -> str:
    """Remove repeated phrases - keep first only"""
    # Split into words
    words = text.split()
    if len(words) < 4:
        return text
    
    # Check for 3+ word sequences repeated
    result = []
    i = 0
    while i < len(words):
        found_repeat = False
        # Check for repeated sequences of 3-6 words
        for seq_len in range(3, min(7, (len(words) - i) // 2 + 1)):
            if i + seq_len * 2 <= len(words):
                seq1 = ' '.join(words[i:i+seq_len])
                seq2 = ' '.join(words[i+seq_len:i+seq_len*2])
                if seq1.lower() == seq2.lower():
                    # Found repeat - keep first only
                    result.extend(words[i:i+seq_len])
                    i += seq_len * 2
                    found_repeat = True
                    break
        if not found_repeat:
            result.append(words[i])
            i += 1
    
    return ' '.join(result)

def apply_rough_cut(segments: List[Segment]) -> List[Segment]:
    """Apply rough cut: remove fillers + repeated phrases"""
    print("🔪 Applying rough cut...")
    
    cleaned_segments = []
    
    for seg in segments:
        # Step 1: Remove filler words
        cleaned_text = remove_filler_words(seg.text)
        
        # Step 2: Remove repeated phrases
        cleaned_text = remove_repeated_phrases(cleaned_text)
        
        # Clean up
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Only keep if meaningful content remains
        if len(cleaned_text) > 15:
            seg.text = cleaned_text
            cleaned_segments.append(seg)
    
    print(f"   Cleaned {len(segments)} segments -> {len(cleaned_segments)} after removing empty")
    
    return cleaned_segments

# ========== END ROUGH CUT ==========

def score_segment(segment: Segment) -> float:
    """Score a segment for viral potential - ENHANCED for interviews"""
    text_lower = segment.text.lower()
    score = 0.0
    reasons = []
    
    # 1. Keyword scoring
    for keyword, weight in VIRAL_KEYWORDS.items():
        count = text_lower.count(keyword)
        if count > 0:
            score += weight * min(count, 3)
            if count == 1:
                reasons.append(keyword)
            else:
                reasons.append(f"{keyword}x{count}")
    
    # 2. ANSWER detection (interview/podcast focus) - BOOST answers
    is_answer = False
    for pattern in ANSWER_INDICATORS:
        if re.search(pattern, segment.text, re.IGNORECASE):
            is_answer = True
            score += 1.5  # Boost for being an answer
            reasons.append("answer")
            break
    
    # 3. QUESTION deprioritization - PENALIZE questions
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, segment.text, re.IGNORECASE):
            score -= 1.5  # Penalize questions
            break
    
    # 4. Duration scoring - prefer 30-60 second clips
    duration = segment.end - segment.start
    if 30 <= duration <= 60:
        score += 2.0  # Sweet spot
    elif duration > 60:
        score += 1.0  # Longer is okay but less ideal
    elif duration >= 15:
        score += 0.5
    elif duration >= 5:
        score += 0.2
    
    # 5. Sentence count - substantial content but not too long
    sentences = len([s for s in re.split(r'[.!?]+', segment.text) if s.strip()])
    if 2 <= sentences <= 6:
        score += 1.2
    
    # 6. CAPS words - indicates emphasis
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', segment.text))
    if 1 <= caps_words <= 3:
        score += 0.8
    
    # 7. Direct statements - "I think", "I believe" are strong
    if re.search(r'\bI think\b', segment.text, re.IGNORECASE):
        score += 1.0
    if re.search(r'\bI believe\b', segment.text, re.IGNORECASE):
        score += 1.0
    if re.search(r'\bmy advice\b', segment.text, re.IGNORECASE):
        score += 2.0
    if re.search(r'\bi recommend\b', segment.text, re.IGNORECASE):
        score += 2.0
    
    segment.score = score
    segment.reason = ", ".join(reasons[:5])
    return score

# Topic change indicators for better segment parsing
TOPIC_CHANGE_PATTERNS = [
    r"^So,", r"^Now,", r"^Moving on", r"^Next,",
    r"^Let me", r"^I want to", r"^Here's", r"^One thing",
    r"^Another", r"^Also,", r"^But also",
    r"\bhowever\b", r"\badditionally\b", r"\bfurthermore\b",
]

def find_topic_changes(segments: List[Segment]) -> List[int]:
    """Find indices of segments that likely indicate topic changes"""
    topic_change_indices = []
    
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        
        # Check for topic change patterns
        for pattern in TOPIC_CHANGE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                topic_change_indices.append(i)
                break
        
        # Also check for long pauses (gap between segments > 30 seconds)
        if i > 0:
            gap = seg.start - segments[i-1].end
            if gap > 30:
                topic_change_indices.append(i)
    
    return topic_change_indices

def enhance_segments_with_topic_parsing(segments: List[Segment]) -> List[Segment]:
    """Enhance segments by splitting/merging at topic changes for better clip boundaries"""
    if not segments:
        return segments
    
    # Find topic change points
    topic_changes = find_topic_changes(segments)
    
    # Mark segments that are at topic changes
    for i in topic_changes:
        if i < len(segments):
            segments[i].score += 0.5  # Slight boost for being at topic change
    
    return segments

def detect_viral_moments(segments: List[Segment], top_n: int = 10, min_duration: int = 30, video_duration: float = 3600) -> List[Segment]:
    """Auto-detect viral moments from transcript segments
    
    Fixed algorithm:
    - Min clip duration: 30 seconds
    - Max clip duration: 60 seconds
    - Spread throughout video (divide into zones)
    - For interview/podcast: focus on ANSWERS not questions
    - NO merging into 4-6 minute clips
    - Better segment parsing for meaningful topic changes
    """
    
    # Apply topic change enhancement
    segments = enhance_segments_with_topic_parsing(segments)
    
    # Config
    MAX_DURATION = 60  # Cap at 60 seconds
    MIN_DURATION = 30  # Minimum 30 seconds
    
    # Score all segments
    for seg in segments:
        score_segment(seg)
    
    # Filter out segments that are too short or too long initially
    filtered = [s for s in segments if 5 <= (s.end - s.start) <= 120]
    
    # Sort by score
    scored_segments = [(s, s.score) for s in filtered]
    scored_segments.sort(key=lambda x: x[1], reverse=True)
    
    # Get top candidates (more candidates for better selection)
    top_candidates = [s for s, _ in scored_segments[:150]]
    top_candidates.sort(key=lambda x: x.start)
    
    # Divide video into zones - aim for 6 clips spread throughout
    num_zones = 6  # Fixed 6 zones for expected 6 clips
    zone_size = video_duration / num_zones
    
    final = []
    used_times = []  # Track used time ranges to avoid overlaps
    
    for zone in range(num_zones):
        zone_start = zone * zone_size
        zone_end = (zone + 1) * zone_size
        
        # Get ALL segments in this zone
        zone_segments = [s for s in top_candidates if zone_start <= s.start < zone_end]
        
        if not zone_segments:
            # If no high-scoring segment, look for any segment in this zone
            zone_segments = [s for s in segments if zone_start <= s.start < zone_end]
        
        if not zone_segments:
            continue
            
        # Sort by score descending
        zone_segments.sort(key=lambda s: s.score, reverse=True)
        
        # Select the BEST segment in this zone
        best_segment = None
        for seg in zone_segments:
            # Check if this segment overlaps with already used time
            overlaps = False
            for used_start, used_end in used_times:
                if not (seg.end <= used_start or seg.start >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                best_segment = seg
                break
        
        if best_segment is None:
            # Find non-overlapping portion or use smallest overlap
            best_segment = zone_segments[0]
        
        # Adjust duration to be within bounds
        duration = best_segment.end - best_segment.start
        
        if duration < MIN_DURATION:
            # Extend backward if possible
            extend_by = MIN_DURATION - duration
            best_segment.start = max(0, best_segment.start - extend_by // 2)
            best_segment.end = min(video_duration, best_segment.end + extend_by // 2)
        
        elif duration > MAX_DURATION:
            # Cap at MAX_DURATION - prefer ending
            best_segment.end = best_segment.start + MAX_DURATION
        
        # Final check - ensure it's still within bounds
        final_duration = best_segment.end - best_segment.start
        if final_duration < MIN_DURATION or final_duration > MAX_DURATION:
            # Skip if still not right
            continue
        
        # Add to final if no major overlap
        has_major_overlap = False
        for used_start, used_end in used_times:
            overlap = min(used_end, best_segment.end) - max(used_start, best_segment.start)
            if overlap > 10:  # More than 10 seconds overlap
                has_major_overlap = True
                break
        
        if not has_major_overlap:
            final.append(best_segment)
            used_times.append((best_segment.start, best_segment.end))
    
    # Sort by timestamp
    final.sort(key=lambda x: x.start)
    
    # Ensure we have exactly 6 clips if possible (or close to top_n)
    if len(final) < top_n:
        # Add more clips from remaining high-scoring segments
        remaining = [s for s in scored_segments if s[0] not in final]
        for seg, score in remaining:
            if len(final) >= top_n:
                break
            # Check overlap
            overlaps = False
            for used_start, used_end in used_times:
                if not (seg.end <= used_start or seg.start >= used_end):
                    overlaps = True
                    break
            if not overlaps:
                # Adjust duration
                duration = seg.end - seg.start
                if duration < MIN_DURATION:
                    seg.start = max(0, seg.start - 5)
                    seg.end = min(video_duration, seg.start + MIN_DURATION)
                elif duration > MAX_DURATION:
                    seg.end = seg.start + MAX_DURATION
                
                if MIN_DURATION <= (seg.end - seg.start) <= MAX_DURATION:
                    final.append(seg)
                    used_times.append((seg.start, seg.end))
    
    return final[:top_n]

def generate_srt(transcript_path: str, output_path: str = None, clip_start: float = None, clip_end: float = None) -> str:
    """Generate SRT subtitle file from transcript with proper timing for a clip.
    
    Args:
        transcript_path: Path to transcript file
        output_path: Path for output SRT file
        clip_start: Start time of clip in original video (seconds). If None, uses absolute timestamps.
        clip_end: End time of clip in original video (seconds)
    """
    segments = parse_transcript(transcript_path)
    
    if not segments:
        print("⚠️ No timestamped segments found in transcript")
        return ""
    
    # Sort by start time
    segments.sort(key=lambda x: x.start)
    
    # If clip bounds provided, filter and adjust timestamps
    if clip_start is not None and clip_end is not None:
        filtered_segments = []
        for seg in segments:
            # Segment overlaps with clip
            if seg.end > clip_start and seg.start < clip_end:
                # Adjust timestamps to be relative to clip
                new_start = max(0, seg.start - clip_start)
                new_end = min(clip_end - clip_start, seg.end - clip_start)
                
                if new_end - new_start >= 1:  # At least 1 second
                    filtered_segments.append(Segment(
                        start=new_start,
                        end=new_end,
                        text=seg.text
                    ))
        segments = filtered_segments
    
    # Break long segments into smaller chunks (2-4 seconds each for subtitles)
    max_segment_duration = 3.5
    min_segment_duration = 2.0
    short_segments = []
    
    for seg in segments:
        duration = seg.end - seg.start
        if duration <= max_segment_duration:
            short_segments.append(seg)
        else:
            # Split into smaller chunks
            num_chunks = max(1, int(duration / max_segment_duration))
            chunk_duration = duration / num_chunks
            
            # Split text by sentences (try to keep sentences together)
            sentences = re.split(r'(?<=[.!?])\s+', seg.text)
            
            current_text = ""
            current_start = seg.start
            chars_per_second = len(seg.text) / duration if duration > 0 else 20
            target_chars = int(chunk_duration * chars_per_second * 0.9)  # 90% to leave room
            
            for i, sentence in enumerate(sentences):
                # If adding this sentence would exceed target and we have content, create a segment
                if len(current_text) + len(sentence) > target_chars and current_text:
                    short_segments.append(Segment(
                        start=current_start,
                        end=current_start + chunk_duration,
                        text=current_text.strip()
                    ))
                    current_start += chunk_duration
                    current_text = sentence
                else:
                    if current_text:
                        current_text += " " + sentence
                    else:
                        current_text = sentence
            
            # Add remaining text
            if current_text.strip():
                short_segments.append(Segment(
                    start=current_start,
                    end=seg.end,
                    text=current_text.strip()
                ))
    
    # Ensure minimum duration for each segment
    final_segments = []
    for seg in short_segments:
        if seg.end - seg.start < min_segment_duration:
            # Extend to minimum duration if possible
            seg.end = seg.start + min_segment_duration
        final_segments.append(seg)
    
    if not final_segments:
        print("⚠️ No valid subtitle segments after processing")
        return ""
    
    # Sort and deduplicate overlapping segments
    final_segments.sort(key=lambda x: x.start)
    merged = []
    for seg in final_segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg.start < last.end - 0.5:  # Overlapping or too close
                # Extend the last segment
                last.end = max(last.end, seg.end)
                last.text = last.text + " " + seg.text
            else:
                merged.append(seg)
    
    srt_path = output_path or transcript_path.replace('.md', '.srt')
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(merged, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n")
            # Clean text for subtitle - remove extra spaces, newlines
            clean_text = re.sub(r'\s+', ' ', seg.text).strip()
            # Limit to ~100 chars per line for readability
            if len(clean_text) > 80 and ' ' in clean_text:
                # Try to break at a natural point
                words = clean_text.split()
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                clean_text = f"{line1}\n{line2}"
            f.write(f"{clean_text}\n\n")
    
    print(f"📝 Subtitles saved: {srt_path} ({len(merged)} segments)")
    return srt_path

def apply_transition(clip, transition: str, duration: float = 0.5) -> "VideoFileClip":
    """Apply transition effect to clip"""
    if transition == "none" or not transition:
        return clip
    
    transition = transition.lower()
    
    if transition == "fade":
        # Fade in and fade out using MoviePy 2.x - function takes (clip, duration)
        clip = vfx.FadeIn(clip, duration)
        clip = vfx.FadeOut(clip, duration)
    elif transition == "dissolve":
        clip = vfx.FadeIn(clip, duration * 1.4)
        clip = vfx.FadeOut(clip, duration * 1.4)
    elif transition in ("wipe-left", "wipe-right", "slide-left", "slide-right"):
        # Use fade as fallback
        clip = vfx.FadeIn(clip, duration)
        clip = vfx.FadeOut(clip, duration)
    elif transition == "zoom":
        clip = vfx.FadeIn(clip, duration * 0.5)
        clip = vfx.FadeOut(clip, duration * 0.5)
    elif transition == "blur":
        clip = vfx.FadeIn(clip, duration)
        clip = vfx.FadeOut(clip, duration)
    
    return clip

def remove_fillers_audio(input_path: str, output_path: str) -> str:
    """Remove silence and create compact audio using ffmpeg"""
    print("🔪 Running rough cut - removing silence...")
    
    # Use silenceremove filter to cut gaps > 0.3 seconds
    # Also normalize audio
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB,areverse,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "copy",  # Keep video unchanged
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Rough cut failed: {result.stderr[:200]}")
        return input_path  # Return original
    
    print(f"✅ Rough cut complete: {output_path}")
    return output_path

def download_video(url: str, output_dir: str, quality: str = "720p") -> str:
    """Download full video using yt-dlp with cookies from Brave"""
    quality_formats = {
        "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]/best",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
    }
    format_str = quality_formats.get(quality, quality_formats["720p"])
    
    output_template = os.path.join(output_dir, "video.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", format_str,
        "-o", output_template,
        "--no-playlist",
        "--cookies-from-browser", "brave",
        url
    ]
    
    print(f"📥 Downloading video ({quality})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for f in os.listdir(output_dir):
        if f.startswith("video.") and not f.endswith(".part"):
            return os.path.join(output_dir, f)
    
    raise FileNotFoundError("Video download failed")

def extract_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> str:
    """Extract a clip using MoviePy - properly handles audio!"""
    print(f"✂️  Extracting clip: {format_duration(start_sec)} - {format_duration(end_sec)}")
    
    with VideoFileClip(video_path) as clip:
        subclip = clip.subclipped(start_sec, end_sec)
        subclip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='fast',
            logger=None
        )
    
    print(f"✅ Saved: {output_path}")
    return output_path

def burn_subtitles(video_path: str, srt_path: str, output_path: str, clip_start: float = 0, clip_end: float = None) -> str:
    """Burn subtitles into video with improved styling and correct timing"""
    if not srt_path or not os.path.exists(srt_path):
        print(f"⚠️ No subtitles file found: {srt_path}")
        return video_path
    
    print(f"🔥 Burning subtitles...")
    
    try:
        import pysubs2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Load subtitles
        subs = pysubs2.load(srt_path)
        
        if not subs:
            print("⚠️ No subtitles loaded from file")
            return video_path
        
        # Get video info
        with VideoFileClip(video_path) as video:
            duration = video.duration
            w, h = video.size
        
        if clip_end is None:
            clip_end = duration
        
        print(f"   Video: {w}x{h}, duration: {duration}s")
        print(f"   Clip: {clip_start}s - {clip_end}s")
        print(f"   Loaded {len(subs)} subtitle lines")
        
        # Create subtitle clips with custom styling
        subtitle_clips = []
        
        for line in subs:
            start_sec = line.start / 1000.0  # Convert ms to seconds
            end_sec = line.end / 1000.0
            
            # Skip if outside clip range
            if end_sec < 0 or start_sec > duration:
                continue
            
            # Clamp to clip bounds
            rel_start = max(0, start_sec)
            rel_end = min(duration, end_sec)
            
            if rel_end - rel_start < 0.3:
                continue
            
            # Clean the subtitle text - remove HTML tags, extra spaces
            text = re.sub(r'<[^>]+>', '', line.text)  # Remove HTML tags
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                continue
            
            # Determine font size based on video dimensions
            # For vertical video (h > w), use larger font
            if h > w:
                font_size = max(36, int(h * 0.045))  # ~4.5% of height
            else:
                font_size = max(28, int(h * 0.035))  # ~3.5% of height
            
            # Create styled subtitle using PIL
            subtitle_img = create_subtitle_image(
                text, 
                video_size=(w, h),
                font_size=font_size,
                text_color='white',
                bg_color=(0, 0, 0, 200),  # Semi-transparent black
                padding=int(font_size * 0.5),
                radius=int(font_size * 0.4),
                margin_bottom=int(h * 0.08)  # 8% from bottom
            )
            
            # Convert PIL to numpy array for MoviePy
            img_array = np.array(subtitle_img)
            
            # Create MoviePy clip from numpy array
            txt_clip = (ImageClip(img_array)
                       .with_duration(rel_end - rel_start)
                       .with_start(rel_start)
                       .with_position(('center', h - subtitle_img.height - int(h * 0.03))))
            subtitle_clips.append(txt_clip)
        
        if not subtitle_clips:
            print("⚠️ No valid subtitles to add")
            return video_path
        
        print(f"   Adding {len(subtitle_clips)} subtitle lines to video...")
        
        # Composite with video
        with VideoFileClip(video_path) as video:
            final = CompositeVideoClip([video] + subtitle_clips)
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                preset='fast',
                logger=None
            )
        
        print(f"✅ Subtitles burned: {output_path}")
        return output_path
        
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")
        return video_path
    except Exception as e:
        print(f"⚠️ Subtitle burning failed: {e}")
        import traceback
        traceback.print_exc()
        return video_path


def create_subtitle_image(text: str, video_size: tuple, font_size: int = 40, 
                         text_color: str = 'white', bg_color: tuple = (0, 0, 0, 200),
                         padding: int = 20, radius: int = 15, margin_bottom: int = 80) -> Image.Image:
    """Create a styled subtitle image with rounded background - dynamically sized to text"""
    from PIL import Image, ImageDraw, ImageFont
    
    vw, vh = video_size
    
    # Try to find a good font - prefer Arial/Helvetica
    font = None
    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/ArialHB.ttc',
        '/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except:
            continue
    
    if font is None:
        # Fall back to default font
        font = ImageFont.load_default()
        # Default font is small, so we need to scale
        font_size = 20
    
    # Wrap text if too long
    max_width = int(vw * 0.85)  # 85% of video width
    lines = wrap_text(text, font, max_width)
    
    # Calculate text size
    line_heights = []
    max_text_width = 0
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_heights.append(h)
        max_text_width = max(max_text_width, w)
    
    # Add padding
    total_width = max_text_width + padding * 2
    total_height = sum(line_heights) + padding * (len(lines) + 1)
    
    # Ensure minimum size
    total_width = max(total_width, 200)
    total_height = max(total_height, font_size + padding * 2)
    
    # Create image with transparency
    img = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle background
    draw.rounded_rectangle(
        [0, 0, total_width - 1, total_height - 1],
        radius=radius,
        fill=bg_color
    )
    
    # Draw text (centered, with proper line spacing)
    y_offset = padding
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (total_width - text_w) // 2
        draw.text((x, y_offset), line, font=font, fill=text_color)
        y_offset += line_heights[i] + 4  # 4px line spacing
    
    return img


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # If still too wide, force break
    if not lines:
        lines = [text]
    
    return lines

def create_reel_with_transitions(clip_paths: list, output_path: str, transition: str = "fade") -> str:
    """Concatenate clips into a reel using ffmpeg (more reliable than MoviePy)"""
    print(f"🎬 Creating reel from {len(clip_paths)} clips...")
    
    # Use ffmpeg for concatenation - more reliable
    concat_file = output_path + ".txt"
    with open(concat_file, 'w') as f:
        for clip_path in clip_paths:
            escaped = clip_path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    
    # Concatenate with ffmpeg
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy",  # Copy streams without re-encoding
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Try with re-encoding
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup
    if os.path.exists(concat_file):
        os.remove(concat_file)
    
    if result.returncode != 0:
        print(f"⚠️ Reel creation failed: {result.stderr[:200]}")
        # Fallback: just copy first clip
        import shutil
        shutil.copy(clip_paths[0], output_path)
    
    print(f"✅ Reel saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Video Repurpose CLI - Auto-detect highlights, add subtitles & transitions")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--clips", "-c", help="Clips to extract (e.g., '0:00-1:00,5:00-6:00')")
    parser.add_argument("--reel", "-r", action="store_true", help="Join clips into a single reel")
    parser.add_argument("--quality", "-q", default=DEFAULT_QUALITY, choices=["360p", "480p", "720p", "1080p"])
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--title", "-t", help="Custom title for output files")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto-detect highlights from transcript")
    parser.add_argument("--transcript", help="Path to transcript file (or auto-detect if not provided)")
    parser.add_argument("--top", "-n", type=int, default=15, help="Number of highlights to extract (default: 15)")
    
    # New features
    parser.add_argument("--rough-cut", action="store_true", help="Remove silence and normalize audio")
    parser.add_argument("--subtitles", "-s", action="store_true", help="Burn in subtitles from transcript")
    parser.add_argument("--transition", "-tr", choices=list(TRANSITIONS.keys()), default="fade", 
                       help="Transition between clips (default: fade)")
    parser.add_argument("--format", "-f", 
                       choices=["youtube", "reels", "tiktok", "shorts", "instagram", "square", "landscape"],
                       default="landscape",
                       help="Output format (default: landscape)")
    parser.add_argument("--crop-mode", 
                       choices=["pad", "crop"],
                       default="pad",
                       help="How to fit video to format: pad=add bars, crop=cut edges")
    
    args = parser.parse_args()
    
    # Show transition options
    if args.transition == "list":
        print("Available transitions:")
        for k, v in TRANSITIONS.items():
            print(f"  {k}: {v}")
        sys.exit(0)
    
    print(f"🔍 Fetching video info...")
    video_info = get_video_info(args.url)
    video_id = video_info['id']
    print(f"📹 {video_info['title']}")
    print(f"⏱️  Duration: {format_duration(video_info['duration'])}")
    
    # Auto-detect mode
    if args.auto or args.transcript:
        if args.transcript:
            transcript_path = args.transcript
        else:
            transcript_path = find_transcript(video_id)
        
        if not transcript_path or not os.path.exists(transcript_path):
            print(f"❌ No transcript found for video ID: {video_id}")
            sys.exit(1)
        
        print(f"📄 Using transcript: {transcript_path}")
        
        # Generate subtitles
        srt_path = ""
        if args.subtitles:
            srt_path = generate_srt(transcript_path)
        
        print(f"🔍 Analyzing transcript for viral moments...")
        segments = parse_transcript(transcript_path)
        print(f"   Found {len(segments)} segments")
        
        highlights = detect_viral_moments(segments, top_n=args.top, video_duration=video_info['duration'])
        
        if not highlights:
            print("❌ No highlights detected.")
            sys.exit(1)
        
        print(f"\n🎯 AUTO-DETECTED HIGHLIGHTS ({len(highlights)} clips):")
        print("=" * 60)
        for i, seg in enumerate(highlights):
            preview = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
            print(f"\n{i+1}. [{format_duration(seg.start)} - {format_duration(seg.end)}] ⭐ {seg.score:.1f}")
            print(f"   {preview}")
        
        args.clips = ",".join([f"{format_duration(s.start)}-{format_duration(s.end)}" for s in highlights])
    
    # Create output directory
    safe_title = args.title or video_info['title'].replace("|", "_").replace("/", "_")[:50]
    output_dir = os.path.join(args.output, safe_title)
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse clips
    if not args.clips:
        print("❌ No clips specified. Use --clips or --auto")
        sys.exit(1)
    
    clip_times = []
    for clip_str in args.clips.split(","):
        clip_str = clip_str.strip()
        if "-" in clip_str:
            start, end = clip_str.split("-")
            start_sec = parse_timestamp(start)
            end_sec = parse_timestamp(end)
            clip_times.append((start_sec, end_sec))
    
    print(f"\n✂️  Will extract {len(clip_times)} clip(s):")
    for i, (s, e) in enumerate(clip_times):
        print(f"   {i+1}. {format_duration(s)} - {format_duration(e)}")
    
    if args.rough_cut:
        print("🔪 Rough cut enabled - will analyze and remove fillers intelligently")
    if args.subtitles:
        print("📝 Subtitles enabled - will burn into video")
    if args.reel:
        print(f"🎬 Transition: {args.transition}")
    
    # Download full video
    video_path = download_video(args.url, output_dir, args.quality)
    print(f"✅ Video downloaded: {video_path}")
    
    # Rough cut - intelligent filler removal
    if args.rough_cut and segments and video_path:
        print("🔪 Analyzing audio and transcript for smart filler removal...")
        try:
            from filler_remover import analyze_and_decide
            
            result = analyze_and_decide(video_path, transcript_path)
            
            print(f"   Mode: {result['mode']}")
            print(f"   {result['summary']}")
            
            if result['mode'] == 'text_only':
                print("   → No clear silence in audio - will clean text only")
                print("   → Subtitles will be generated without filler words")
            elif result['mode'] == 'audio_cut':
                print("   → Found silence in audio - will cut both audio and text")
            
            # Use cleaned transcript for subtitles if available
            if result.get('cleaned_transcript'):
                # Save cleaned transcript
                cleaned_srt = os.path.join(output_dir, "cleaned.srt")
                with open(cleaned_srt, 'w') as f:
                    f.write(result['cleaned_transcript'])
                print(f"   ✅ Cleaned transcript saved: {cleaned_srt}")
                
        except Exception as e:
            print(f"   ⚠️ Smart filler detection failed: {e}")
            print("   → Falling back to basic cleanup")
            segments = apply_rough_cut(segments)
    
    # Extract clips
    clip_paths = []
    for i, (start_sec, end_sec) in enumerate(clip_times):
        clip_name = f"clip_{i+1}_{format_duration(start_sec).replace(':', '-')}-{format_duration(end_sec).replace(':', '-')}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        extract_clip(video_path, start_sec, end_sec, clip_path)
        
        # Add subtitles if requested
        if args.subtitles and transcript_path:
            # Generate clip-specific subtitles with proper timing
            # Pass clip_start and clip_end to adjust timestamps correctly
            clip_srt = generate_srt(
                transcript_path, 
                output_path=clip_path.replace('.mp4', '.srt'),
                clip_start=start_sec,
                clip_end=end_sec
            )
            
            if clip_srt and os.path.exists(clip_srt):
                # Burn subtitles with correct timing
                subbed_path = clip_path.replace('.mp4', '_subbed.mp4')
                clip_path = burn_subtitles(clip_path, clip_srt, subbed_path, clip_start=0, clip_end=end_sec - start_sec)
        
        clip_paths.append(clip_path)
    
    # Create reel with transitions
    if args.reel and len(clip_paths) > 1:
        reel_path = os.path.join(output_dir, f"reel_{args.transition}.mp4")
        create_reel_with_transitions(clip_paths, reel_path, args.transition)
        
        # Add subtitles to final reel if not added per-clip
        if args.subtitles and not args.transcript:
            final_reel = reel_path.replace('.mp4', '_subbed.mp4')
            if srt_path:
                reel_path = burn_subtitles(reel_path, srt_path, final_reel)
        
        print(f"🎬 Reel created: {reel_path}")
    elif args.reel and len(clip_paths) == 1:
        print("ℹ️  Only 1 clip - no reel needed")
    
    # Convert to target format if not landscape
    if args.format != "landscape" and args.format != "youtube":
        from video_format import convert_format
        
        if 'reel_path' in locals():
            output_video = reel_path
        elif clip_paths:
            output_video = clip_paths[0]
        else:
            output_video = None
        
        if output_video and os.path.exists(output_video):
            formatted_path = output_video.replace(".mp4", f"_{args.format}.mp4")
            print(f"📐 Converting to {args.format} format...")
            convert_format(output_video, formatted_path, args.format, mode=args.crop_mode)
            print(f"✅ Formatted: {formatted_path}")
    
    # Cleanup
    if os.path.exists(video_path) and video_path != clip_paths[0]:
        try:
            os.remove(video_path)
        except:
            pass
    
    print(f"\n✅ Done! Files saved to: {output_dir}")
    print(f"📁 Files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".mp4"):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
            print(f"   - {f} ({size:.1f}MB)")

if __name__ == "__main__":
    main()
