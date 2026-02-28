#!/usr/bin/env python3
"""
Speaker Diarization Module using pyannote.audio

Detects and labels speakers in video or audio files.
Returns speaker segments with timestamps labeled as SPEAKER_1, SPEAKER_2, etc.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# pyannote.audio imports are lazy-loaded inside get_pipeline()
# so this module can be imported without pyannote installed


# Dataclass for speaker segments
class SpeakerSegment:
    """Represents a speaker segment with timestamp"""
    def __init__(self, start: float, end: float, speaker: str, confidence: float = 1.0):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.confidence = confidence
    
    def __repr__(self):
        return f"SpeakerSegment(start={self.start:.2f}, end={self.end:.2f}, speaker={self.speaker}, confidence={self.confidence:.2f})"
    
    def __str__(self):
        return f"[{self.start:.2f} - {self.end:.2f}] {self.speaker} ({self.confidence:.2f})"


# Cache for the pipeline (to avoid reloading model each time)
_pipeline = None


def get_pipeline() -> Pipeline:
    """Get or create the pyannote diarization pipeline (cached)"""
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline
    
    print("🔊 Loading speaker diarization model...")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            import huggingface_hub
            hf_token = huggingface_hub.get_token()
        except Exception:
            pass
    
    # Try WhisperX first (doesn't require gated model access)
    try:
        import whisperx
        if not hasattr(whisperx, 'DiarizationPipeline'):
            raise AttributeError("WhisperX installed but DiarizationPipeline not available")
        print("✅ Using WhisperX for speaker diarization")
        _pipeline = "whisperx"
        return _pipeline
    except (ImportError, AttributeError) as e:
        print(f"⚠️ WhisperX not usable: {e}")
        pass
    
    # Shim for torchaudio 2.10+ which removed list_audio_backends
    try:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['ffmpeg']
    except ImportError:
        pass

    # Lazy import pyannote
    from pyannote.audio import Pipeline

    # Try loading with token first (pyannote)
    if hf_token:
        try:
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            print("✅ Speaker diarization model loaded (v3.1 with token)")
            return _pipeline
        except Exception as e:
            print(f"⚠️ v3.1 failed: {e}")

    # Try pyannote v2.1
    try:
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-2.1")
        print("✅ Speaker diarization model loaded (v2.1)")
    except Exception as e:
        print(f"⚠️ v2.1 failed: {e}")
        print("\n💡 Install whisperx for easier setup: pip install whisperx")
        print("   Or set HF_TOKEN: https://huggingface.co/settings/tokens")
        raise RuntimeError("Cannot load speaker diarization model")


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file using ffmpeg"""
    print(f"🎬 Extracting audio from video: {video_path}")
    
    # Create temp file for audio
    audio_path = tempfile.mktemp(suffix=".wav")
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16kHz sample rate (required by pyannote)
        "-ac", "1",  # Mono
        audio_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")
    
    print(f"✅ Audio extracted: {audio_path}")
    return audio_path


def detect_speakers(
    audio_path: str, 
    min_speakers: int = 1, 
    max_speakers: int = 10,
    video_path: Optional[str] = None
) -> List[SpeakerSegment]:
    """
    Detect speakers in audio file.
    
    Args:
        audio_path: Path to audio file (or video file)
        min_speakers: Minimum number of expected speakers
        max_speakers: Maximum number of expected speakers
        video_path: Optional video path (if audio_path is actually video)
    
    Returns:
        List of SpeakerSegment objects with timestamps and speaker labels
    """
    # If video provided, extract audio
    actual_audio_path = audio_path
    
    # Check if it's a video file
    video_extensions = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv'}
    file_ext = Path(audio_path).suffix.lower()
    
    if file_ext in video_extensions or video_path:
        actual_audio_path = extract_audio_from_video(video_path or audio_path)
    
    if not os.path.exists(actual_audio_path):
        raise FileNotFoundError(f"Audio file not found: {actual_audio_path}")
    
    print(f"🔍 Running speaker diarization...")
    
    # Run diarization
    pipeline = get_pipeline()
    
    # Apply pipeline
    if pipeline == "whisperx":
        import whisperx
        diarization = whisperx.DiarizationPipeline()(
            actual_audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    else:
        diarization = pipeline(
            actual_audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    
    # Convert to our segment format
    # pyannote v4 returns DiarizeOutput wrapper; v3 returns Annotation directly
    if hasattr(diarization, 'speaker_diarization'):
        diarization = diarization.speaker_diarization

    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
            confidence=1.0  # pyannote doesn't provide per-segment confidence
        ))
    
    # Clean up temp audio file if we created one
    if actual_audio_path != audio_path and os.path.exists(actual_audio_path):
        try:
            os.remove(actual_audio_path)
        except:
            pass
    
    print(f"✅ Detected {len(set(s.speaker for s in segments))} speakers, {len(segments)} segments")
    
    return segments


def get_main_speaker(segments: List[SpeakerSegment]) -> str:
    """
    Find the main speaker (the one who speaks the most total time).
    
    Args:
        segments: List of speaker segments
    
    Returns:
        Speaker label of the main speaker (e.g., "SPEAKER_00")
    """
    speaker_times = {}
    
    for seg in segments:
        if seg.speaker not in speaker_times:
            speaker_times[seg.speaker] = 0
        speaker_times[seg.speaker] += (seg.end - seg.start)
    
    # Find speaker with most speaking time
    main_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
    
    return main_speaker


def filter_main_speaker_segments(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
    """
    Filter segments to only include the main speaker.
    
    Args:
        segments: List of all speaker segments
    
    Returns:
        List of segments from the main speaker only
    """
    if not segments:
        return []
    
    main_speaker = get_main_speaker(segments)
    print(f"🎤 Main speaker identified: {main_speaker}")
    
    # Calculate speaking times
    speaker_times = {}
    for seg in segments:
        if seg.speaker not in speaker_times:
            speaker_times[seg.speaker] = 0
        speaker_times[seg.speaker] += (seg.end - seg.start)
    
    # Print all speakers with their total speaking time
    print("\n📊 Speaker speaking times:")
    for speaker, total_time in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True):
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"   {speaker}: {minutes}m {seconds}s")
    
    # Filter to main speaker only
    main_segments = [seg for seg in segments if seg.speaker == main_speaker]
    
    print(f"\n✅ Filtered to {len(main_segments)} segments from {main_speaker}")
    
    return main_segments


def merge_adjacent_segments(segments: List[SpeakerSegment], gap_threshold: float = 0.5) -> List[SpeakerSegment]:
    """
    Merge adjacent segments from the same speaker if gap is small.
    
    Args:
        segments: List of speaker segments
        gap_threshold: Maximum gap in seconds to merge
    
    Returns:
        Merged segments
    """
    if not segments:
        return []
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x.start)
    
    merged = [sorted_segments[0]]
    
    for seg in sorted_segments[1:]:
        last = merged[-1]
        
        # If same speaker and gap is small, merge
        if seg.speaker == last.speaker and seg.start - last.end < gap_threshold:
            last.end = seg.end
        else:
            merged.append(seg)
    
    return merged


def get_speaker_timeline(segments: List[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Get timeline of each speaker as dict of speaker -> list of (start, end) tuples.
    
    Args:
        segments: List of speaker segments
    
    Returns:
        Dict mapping speaker labels to their timeline segments
    """
    timeline = {}
    
    for seg in segments:
        if seg.speaker not in timeline:
            timeline[seg.speaker] = []
        timeline[seg.speaker].append((seg.start, seg.end))
    
    return timeline


def detect_speakers_from_video(
    video_path: str,
    min_speakers: int = 1,
    max_speakers: int = 10,
    main_speaker_only: bool = False,
    merge_gaps: bool = True
) -> List[SpeakerSegment]:
    """
    Convenience function to detect speakers directly from video file.
    
    Args:
        video_path: Path to video file
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers
        main_speaker_only: If True, return only main speaker segments
        merge_gaps: If True, merge adjacent segments from same speaker
    
    Returns:
        List of SpeakerSegment objects
    """
    print(f"\n🎬 Processing video: {video_path}")
    print(f"   Min speakers: {min_speakers}, Max speakers: {max_speakers}")
    
    # Detect speakers
    segments = detect_speakers(
        audio_path=video_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    if not segments:
        print("⚠️ No speaker segments detected")
        return []
    
    # Print summary
    print(f"\n📋 Speaker Detection Summary:")
    print(f"   Total segments: {len(segments)}")
    unique_speakers = set(seg.speaker for seg in segments)
    print(f"   Unique speakers: {len(unique_speakers)}")
    
    # Print sample segments
    print(f"\n📝 Sample segments (first 10):")
    for seg in segments[:10]:
        print(f"   {seg}")
    if len(segments) > 10:
        print(f"   ... and {len(segments) - 10} more segments")
    
    # Filter to main speaker only if requested
    if main_speaker_only:
        segments = filter_main_speaker_segments(segments)
    
    # Merge gaps if requested
    if merge_gaps:
        segments = merge_adjacent_segments(segments)
    
    return segments


def segments_to_speaker_dict(segments: List[SpeakerSegment]) -> Dict[str, List[Tuple[float, float]]]:
    """Convert list of SpeakerSegments to {label: [(start, end), ...]} dict format."""
    result = {}
    for seg in segments:
        if seg.speaker not in result:
            result[seg.speaker] = []
        result[seg.speaker].append((seg.start, seg.end))
    return result


def diarize_to_json(audio_path: str, output_path: str, min_speakers: int = 1, max_speakers: int = 10):
    """Run diarization and write results as JSON for inter-process communication."""
    import json
    segments = detect_speakers(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
    merged = merge_adjacent_segments(segments)

    output = {
        "speakers": [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker}
            for seg in merged
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Diarization saved to {output_path}")
    return output


# Test function
def test_speaker_detection():
    """Test speaker detection with a video file"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test speaker detection")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--main-only", action="store_true", help="Show main speaker only")
    parser.add_argument("--merge", action="store_true", default=True, help="Merge adjacent segments")
    
    args = parser.parse_args()
    
    segments = detect_speakers_from_video(
        args.video_path,
        main_speaker_only=args.main_only,
        merge_gaps=args.merge
    )
    
    print(f"\n🎯 Final segments: {len(segments)}")
    for seg in segments:
        print(f"   {seg}")
    
    return segments


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speaker_detector.py <video_path> [--main-only] [--merge]")
        sys.exit(1)
    
    test_speaker_detection()
