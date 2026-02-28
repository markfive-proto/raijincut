"""
Filler Remover Integration Module
Combines audio and transcript analysis to make intelligent filler removal decisions.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import the analysis modules
from audio_analysis import detect_silence, get_audio_info
from transcript_filler import detect_fillers, generate_cleaned_srt, parse_srt


@dataclass
class CutDecision:
    """Represents a decision about whether to cut/remove a segment."""
    start: float
    end: float
    action: str  # "remove" | "shorten" | "keep" | "review"
    confidence: float  # 0.0 to 1.0
    reason: str
    has_silence: bool
    has_filler: bool


class FillerRemover:
    """
    Integrates audio silence detection and transcript filler detection
    to make intelligent decisions about what to remove.
    """
    
    def __init__(
        self,
        min_silence_duration: float = 0.3,
        silence_threshold_db: float = -40,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the filler remover.
        
        Args:
            min_silence_duration: Minimum silence duration in seconds
            silence_threshold_db: Noise level threshold in dB
            confidence_threshold: Minimum confidence to auto-decide
        """
        self.min_silence_duration = min_silence_duration
        self.silence_threshold_db = silence_threshold_db
        self.confidence_threshold = confidence_threshold
    
    def analyze(
        self,
        video_path: str,
        transcript_path: str
    ) -> Dict:
        """
        Analyze video and transcript to determine what to remove.
        
        Args:
            video_path: Path to video file
            transcript_path: Path to SRT transcript
        
        Returns:
            Dictionary with analysis results including:
            - mode: "audio_cut" | "text_only" | "none"
            - cuts: List of (start, end, action) tuples
            - cleaned_transcript: SRT with fillers removed
        """
        # Detect silence in audio
        silence_gaps = detect_silence(
            video_path,
            min_silence=self.min_silence_duration,
            noise_level=self.silence_threshold_db
        )
        
        # Detect fillers in transcript
        fillers = detect_fillers(transcript_path)
        
        # Determine mode based on audio characteristics
        has_significant_silence = len(silence_gaps) > 0 and any(s[2] > 0.3 for s in silence_gaps)
        
        if has_significant_silence:
            # Mode 1: Audio has clear silence - we can cut audio
            mode = "audio_cut"
        elif len(fillers) > 0:
            # Mode 2: No clear silence but has fillers - text-only cleaning
            mode = "text_only"
        else:
            # Mode 3: Nothing to do
            mode = "none"
        
        # Make decisions
        decisions = self._make_decisions(silence_gaps, fillers, has_significant_silence)
        
        # Generate outputs based on mode
        if mode == "audio_cut":
            cuts = self._generate_cuts(decisions)
            cleaned_transcript = generate_cleaned_srt(
                transcript_path,
                [f for f in fillers if self._should_remove(f, decisions)]
            )
        elif mode == "text_only":
            # For text-only mode, only remove from transcript, don't cut audio
            cuts = []  # No audio cuts
            cleaned_transcript = generate_cleaned_srt(
                transcript_path,
                [f for f in fillers if f['word'] in {'you know', 'um', 'uh', 'er', 'ah'}]
            )
        else:
            cuts = []
            cleaned_transcript = None
        
        # Count stats
        stats = self._count_stats(decisions)
        stats['mode'] = mode
        
        return {
            'mode': mode,
            'cuts': cuts,
            'cleaned_transcript': cleaned_transcript,
            'summary': self._generate_summary(stats, mode),
            'decisions': decisions,
            'stats': stats,
            'silence_gaps': silence_gaps,
            'fillers': fillers
        }
    
    def _make_decisions(
        self,
        silence_gaps: List[Tuple[float, float, float]],
        fillers: List[Dict],
        has_significant_silence: bool = True
    ) -> List[CutDecision]:
        """
        Make intelligent decisions about each filler segment.
        
        Decision logic:
        - Silence + Filler = High confidence remove
        - Silence + No filler = Maybe natural pause (keep or shorten)
        - Filler + No silence = Analyze filler type for decision
        - Neither = Keep
        """
        decisions = []
        
        # High-confidence filler words (safe to remove even without silence)
        high_confidence_fillers = {'you know', 'like', 'kind of', 'sort of', 'basically', 
                                   'actually', 'honestly', 'literally', 'anyway', 'i mean',
                                   'kinda', 'sorta', 'you see', 'i guess'}
        
        # Medium-confidence filler words
        medium_confidence_fillers = {'i think', 'maybe', 'perhaps', 'so', 
                                     'well', 'right', 'yeah', 'yep', 'nope', 'sure',
                                     "i don't know", 'as i mentioned', 'as i said'}
        
        has_silence = len(silence_gaps) > 0
        
        # Make decision for each filler
        for filler in fillers:
            filler_start = filler['start_seconds']
            filler_end = filler['end_seconds']
            filler_word = filler['word']
            
            # Check for nearby silence
            nearby_silence = self._find_nearby_silence(
                filler_start, filler_end, silence_gaps
            )
            
            has_nearby_silence = nearby_silence is not None
            has_filler = True
            
            # Make decision based on combination
            if has_nearby_silence:
                # High confidence: silence + filler = remove
                confidence = 0.95
                action = "remove"
                reason = f"Silence detected ({nearby_silence[2]:.1f}s gap) at filler location"
            
            elif not has_silence:
                # No silence in entire audio - make decision based on filler type
                if filler_word in high_confidence_fillers:
                    confidence = 0.8
                    action = "remove"
                    reason = f"Common filler '{filler_word}' - safe to remove (no silence in audio)"
                elif filler_word in medium_confidence_fillers:
                    confidence = 0.6
                    action = "shorten"
                    reason = f"Moderate filler '{filler_word}' - can shorten"
                else:
                    # Unknown filler - mark for review
                    confidence = 0.4
                    action = "review"
                    reason = f"Unknown filler '{filler_word}' - needs review"
            else:
                # Has silence but this filler doesn't align with it
                confidence = 0.5
                
                # Check if filler is at sentence boundaries (might be natural)
                if filler['context'].strip().endswith(('.', '!', '?')):
                    action = "keep"
                    reason = "Filler at sentence end - possibly natural"
                    confidence = 0.7
                else:
                    # Try to remove common fillers even without nearby silence
                    if filler_word in high_confidence_fillers:
                        action = "remove"
                        confidence = 0.7
                        reason = f"Common filler '{filler_word}'"
                    else:
                        action = "review"
                        reason = "Filler detected but no clear silence - needs human review"
            
            decision = CutDecision(
                start=filler_start,
                end=filler_end,
                action=action,
                confidence=confidence,
                reason=reason,
                has_silence=has_nearby_silence,
                has_filler=has_filler
            )
            decisions.append(decision)
        
        # Also analyze silence gaps without fillers
        for start, end, duration in silence_gaps:
            # Check if there's a filler in this gap
            has_filler_in_gap = any(
                f['start_seconds'] >= start and f['end_seconds'] <= end
                for f in fillers
            )
            
            if not has_filler_in_gap and duration > 0.5:
                # Significant silence without filler
                if duration > 1.0:
                    action = "shorten"
                    confidence = 0.7
                    reason = f"Long silence ({duration:.1f}s) without filler - shorten"
                else:
                    action = "keep"
                    confidence = 0.6
                    reason = f"Short silence ({duration:.1f}s) - possibly natural pause"
                
                decisions.append(CutDecision(
                    start=start,
                    end=end,
                    action=action,
                    confidence=confidence,
                    reason=reason,
                    has_silence=True,
                    has_filler=False
                ))
        
        # Sort by start time
        decisions.sort(key=lambda d: d.start)
        
        return decisions
    
    def _find_nearby_silence(
        self,
        filler_start: float,
        filler_end: float,
        silence_gaps: List[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find silence gap that overlaps or is near the filler."""
        tolerance = 0.5  # seconds
        
        for start, end, duration in silence_gaps:
            # Check for overlap
            if (start <= filler_end + tolerance and end >= filler_start - tolerance):
                return (start, end, duration)
        
        return None
    
    def _time_bucket(self, time: float, bucket_size: float = 1.0) -> int:
        """Round time to bucket for efficient lookup."""
        return int(time // bucket_size)
    
    def _should_remove(self, filler: Dict, decisions: List[CutDecision]) -> bool:
        """Check if a filler should be removed based on decisions."""
        for d in decisions:
            if (abs(d.start - filler['start_seconds']) < 0.5 and
                d.action in ("remove", "shorten")):
                return True
        return False
    
    def _generate_cuts(self, decisions: List[CutDecision]) -> List[Tuple[float, float, str]]:
        """Generate cut list from decisions."""
        cuts = []
        for d in decisions:
            if d.action in ("remove", "shorten"):
                end_time = d.end
                if d.action == "shorten":
                    # Shorten by 50%
                    end_time = d.start + (d.end - d.start) * 0.5
                cuts.append((d.start, round(end_time, 3), d.action))
        
        return cuts
    
    def _count_stats(self, decisions: List[CutDecision]) -> Dict:
        """Count statistics about decisions."""
        stats = {
            'total_fillers': sum(1 for d in decisions if d.has_filler),
            'total_silences': sum(1 for d in decisions if d.has_silence),
            'remove': sum(1 for d in decisions if d.action == "remove"),
            'shorten': sum(1 for d in decisions if d.action == "shorten"),
            'keep': sum(1 for d in decisions if d.action == "keep"),
            'review': sum(1 for d in decisions if d.action == "review"),
        }
        return stats
    
    def _generate_summary(self, stats: Dict, mode: str = "none") -> str:
        """Generate human-readable summary."""
        mode_descriptions = {
            "audio_cut": "Audio + Text mode: Cutting silence and removing fillers",
            "text_only": "Text-only mode: Removing fillers from transcript only (no audio cutting)",
            "none": "No action needed"
        }
        
        parts = [mode_descriptions.get(mode, mode)]
        
        if stats.get('total_fillers', 0) > 0:
            parts.append(f"{stats['total_fillers']} fillers found")
        
        if stats.get('remove', 0) > 0:
            parts.append(f"{stats['remove']} will be removed")
        
        if stats.get('shorten', 0) > 0:
            parts.append(f"{stats['shorten']} will be shortened")
        
        if stats.get('review', 0) > 0:
            parts.append(f"{stats['review']} need review")
        
        return " | ".join(parts)


def analyze_and_decide(
    video_path: str,
    transcript_path: str,
    min_silence: float = 0.3,
    silence_threshold: float = -40
) -> Dict:
    """
    Convenience function to analyze and decide what to remove.
    
    Args:
        video_path: Path to video file
        transcript_path: Path to SRT transcript
        min_silence: Minimum silence duration in seconds
        silence_threshold: Silence detection threshold in dB
    
    Returns:
        Dictionary with:
        - cuts: List of (start, end, action) tuples
        - cleaned_transcript: SRT string with fillers removed
        - summary: Human-readable summary string
    """
    remover = FillerRemover(
        min_silence_duration=min_silence,
        silence_threshold_db=silence_threshold
    )
    
    result = remover.analyze(video_path, transcript_path)
    
    return {
        'mode': result.get('mode', 'unknown'),
        'cuts': result['cuts'],
        'cleaned_transcript': result['cleaned_transcript'],
        'summary': result['summary']
    }


if __name__ == "__main__":
    # Test with command line arguments
    if len(sys.argv) < 3:
        print("Usage: python filler_remover.py <video_path> <transcript_path>")
        print("\nExample:")
        print("  python filler_remover.py video.webm transcript.srt")
        sys.exit(1)
    
    video_path = sys.argv[1]
    transcript_path = sys.argv[2]
    
    print(f"Analyzing:")
    print(f"  Video: {video_path}")
    print(f"  Transcript: {transcript_path}")
    print("-" * 60)
    
    result = analyze_and_decide(video_path, transcript_path)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSummary: {result['summary']}")
    print(f"\nCuts to make: {len(result['cuts'])}")
    for start, end, action in result['cuts']:
        print(f"  [{start:.1f}s -> {end:.1f}s] {action}")
    
    print(f"\nCleaned transcript ({len(result['cleaned_transcript'])} chars):")
    print("-" * 40)
    print(result['cleaned_transcript'][:1000])
    if len(result['cleaned_transcript']) > 1000:
        print("... (truncated)")
