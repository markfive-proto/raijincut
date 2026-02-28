"""
Transcript Filler Detection Module
Detects filler words in SRT transcript files.
"""

import re
from typing import List, Tuple, Dict
from pathlib import Path


# Common filler words and phrases to detect
FILLER_PATTERNS = [
    # Um, uh, ah, er
    r'\b(?:um|uh|ah|er|eh)\b',
    # Like (filler usage)
    r'\blike\b(?=\s+(?:I|you|he|she|we|they|it|a|an|the|to|for|and|but|or|so|just|really|very|actually|kinda|sorta|maybe|you know|I mean|kind of|sort of))',
    # You know
    r'\byou know\b',
    # I mean
    r'\bi mean\b',
    # Kind of / sort of
    r'\b(?:kind of|sort of|kinda|sorta)\b',
    # Basically
    r'\bbasically\b',
    # Actually
    r'\bactually\b',
    # Really (as filler)
    r'\breally\b(?=\s+(?:\w{1,5}\b))',
    # Very (as filler)
    r'\bvery\b(?=\s+(?:\w{1,5}\b))',
    # So (as filler at sentence start)
    r'^\s*so\b(?=\s+(?:I|you|he|she|we|they|it|the|a|an|what|how|why|when|where))',
    # Well (as filler)
    r'\bwell\b(?=\s+(?:I|you|he|she|we|they|it|the|a|an|as|you know|now))',
    # Right (as filler)
    r'\bright\b(?=\s*,|\s+so|\s+then|\s+then\b)',
    # Okay / OK (as filler)
    r'\b(?:okay|ok|OK)\b(?=\s+[,\.]|\s+[IWYiwyi][\s,]|$)',
    # I guess
    r'\bi guess\b',
    # I think
    r'\bi think\b(?=\s*,|\s+that|\s+it|\s+the)',
    # Maybe / perhaps
    r'\bmaybe\b',
    r'\bperhaps\b',
    # You see
    r'\byou see\b',
    # I suppose
    r'\bi suppose\b',
    # As I said / as I mentioned
    r'\bas I (?:said|mentioned|was saying)\b',
    # Let me see / let me think
    r'\blet me (?:see|think|check)\b',
    # You know what I mean
    r'\byou know what I mean\b',
    # Anyway / anyways
    r'\b(?:anyway|anyways)\b',
    # Literally (overuse)
    r'\bliterally\b',
    # Honestly
    r'\bhonestly\b',
    # Basically
    r'\bbasically\b',
    # Obviously
    r'\bobviously\b',
    # Sure (as filler)
    r'\bsure\b(?=\s*,)',
    # Yeah / yep / nope
    r'\b(?:yeah|yep|nope|yah)\b(?=\s*[,\.]|\s+[,\.]|$)',
    # I don't know (as filler)
    r"\bi don't know\b(?=\s*[,.]|\s+[A-Z]|$)",
]


def parse_srt(file_path: str) -> List[Dict]:
    """
    Parse an SRT file into a list of subtitle entries.
    
    Args:
        file_path: Path to the SRT file
    
    Returns:
        List of dictionaries with 'index', 'start_time', 'end_time', 'text'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match SRT entries
    # 1\n00:00:00,000 --> 00:00:05,000\nText\n\n
    pattern = re.compile(
        r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\d+\n\d{2}:\d{2}:\d{2},\d{3}).)*)',
        re.DOTALL
    )
    
    subtitles = []
    for match in pattern.finditer(content):
        index = int(match.group(1))
        start_time = match.group(2)
        end_time = match.group(3)
        text = match.group(4).strip()
        
        subtitles.append({
            'index': index,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'start_seconds': time_to_seconds(start_time),
            'end_seconds': time_to_seconds(end_time)
        })
    
    return subtitles


def time_to_seconds(time_str: str) -> float:
    """
    Convert SRT timestamp (HH:MM:SS,mmm) to seconds.
    
    Args:
        time_str: Timestamp in format HH:MM:SS,mmm
    
    Returns:
        Time in seconds as float
    """
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Timestamp in format HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def detect_fillers(
    file_path: str,
    custom_patterns: List[str] = None
) -> List[Dict]:
    """
    Detect filler words in an SRT transcript.
    
    Args:
        file_path: Path to the SRT file
        custom_patterns: Optional list of additional regex patterns
    
    Returns:
        List of dictionaries with filler info: 
        {
            'word': str,
            'start_seconds': float,
            'end_seconds': float,
            'subtitle_index': int,
            'context': str
        }
    """
    subtitles = parse_srt(file_path)
    
    # Combine all patterns
    all_patterns = FILLER_PATTERNS.copy()
    if custom_patterns:
        all_patterns.extend(custom_patterns)
    
    # Compile patterns
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in all_patterns]
    
    fillers = []
    
    for subtitle in subtitles:
        text = subtitle['text']
        
        for pattern in compiled_patterns:
            for match in pattern.finditer(text):
                filler_word = match.group(0)
                
                # Estimate position within subtitle for timing
                match_start = match.start()
                text_before = text[:match_start]
                text_length = len(text)
                
                # Approximate timing based on character position
                ratio = match_start / max(text_length, 1)
                subtitle_duration = subtitle['end_seconds'] - subtitle['start_seconds']
                filler_start = subtitle['start_seconds'] + (ratio * subtitle_duration)
                filler_end = filler_start + (0.3 if len(filler_word) < 3 else 0.5)  # Estimate duration
                
                fillers.append({
                    'word': filler_word.lower(),
                    'start_seconds': round(filler_start, 3),
                    'end_seconds': round(min(filler_end, subtitle['end_seconds']), 3),
                    'subtitle_index': subtitle['index'],
                    'context': text[:100]  # First 100 chars for context
                })
    
    # Sort by start time
    fillers.sort(key=lambda x: x['start_seconds'])
    
    # Remove duplicates (same word in same time range)
    seen = set()
    unique_fillers = []
    for f in fillers:
        key = (f['word'], round(f['start_seconds'], 1))
        if key not in seen:
            seen.add(key)
            unique_fillers.append(f)
    
    return unique_fillers


def generate_cleaned_srt(
    original_path: str,
    fillers_to_remove: List[Dict],
    output_path: str = None
) -> str:
    """
    Generate a cleaned SRT with specified fillers removed.
    
    Args:
        original_path: Path to original SRT file
        fillers_to_remove: List of filler dicts to remove
        output_path: Optional output path (if None, returns string)
    
    Returns:
        Cleaned SRT content or path to saved file
    """
    subtitles = parse_srt(original_path)
    
    # Create a set of times to remove (with small buffer)
    remove_ranges = set()
    for f in fillers_to_remove:
        # Mark this filler for removal
        remove_ranges.add((
            round(f['start_seconds'], 3),
            round(f['end_seconds'], 3)
        ))
    
    cleaned_subtitles = []
    
    for subtitle in subtitles:
        text = subtitle['text']
        
        # Remove filler words from text
        for pattern in FILLER_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation that might look weird
        text = text.strip('.,;:')
        
        # Only keep subtitle if it still has content
        if text:
            subtitle['text'] = text
            cleaned_subtitles.append(subtitle)
    
    # Build output
    output = []
    for i, sub in enumerate(cleaned_subtitles, 1):
        output.append(str(i))
        output.append(f"{sub['start_time']} --> {sub['end_time']}")
        output.append(sub['text'])
        output.append('')
    
    content = '\n'.join(output)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path
    
    return content


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcript_filler.py <transcript.srt>")
        sys.exit(1)
    
    transcript_path = sys.argv[1]
    
    print(f"Detecting fillers in: {transcript_path}")
    print("-" * 50)
    
    fillers = detect_fillers(transcript_path)
    
    if fillers:
        print(f"Found {len(fillers)} filler instances:\n")
        
        # Group by filler word
        from collections import Counter
        word_counts = Counter(f['word'] for f in fillers)
        
        print("Filler word frequency:")
        for word, count in word_counts.most_common():
            print(f"  {word}: {count}")
        
        print("\nTimeline:")
        for f in fillers[:20]:  # Show first 20
            print(f"  [{f['start_seconds']:.1f}s - {f['end_seconds']:.1f}s] {f['word']}")
        
        if len(fillers) > 20:
            print(f"  ... and {len(fillers) - 20} more")
    else:
        print("No fillers detected.")
