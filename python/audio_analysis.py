"""
Audio Waveform Analysis Module
Detects silence in video/audio files using ffmpeg's silencedetect filter.
"""

import subprocess
import re
from typing import List, Tuple


def detect_silence(
    file_path: str,
    min_silence: float = 0.3,
    noise_level: float = -40
) -> List[Tuple[float, float, float]]:
    """
    Detect silence in a video or audio file.
    
    Args:
        file_path: Path to the video/audio file
        min_silence: Minimum silence duration in seconds (default: 0.3)
        noise_level: Noise level in dB (default: -40)
    
    Returns:
        List of tuples (start_time, end_time, silence_duration) for each silence gap
    """
    # Build the ffmpeg command
    # silencedetect returns: silence_start: X, silence_end: Y | duration: Z
    cmd = [
        'ffmpeg',
        '-i', file_path,
        '-af', f'silencedetect=noise={noise_level}dB:d={min_silence}',
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Parse stderr for silence detection output
        output = result.stderr
        
        silence_gaps = []
        silence_start = None
        
        # Pattern to match silence_start timestamps
        start_pattern = re.compile(r'silence_start:\s*([\d.]+)')
        # Pattern to match silence_end timestamps
        end_pattern = re.compile(r'silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)')
        
        for line in output.split('\n'):
            # Look for silence_start
            start_match = start_pattern.search(line)
            if start_match:
                silence_start = float(start_match.group(1))
            
            # Look for silence_end
            end_match = end_pattern.search(line)
            if end_match and silence_start is not None:
                silence_end = float(end_match.group(1))
                silence_duration = float(end_match.group(2))
                
                # Only include if duration meets minimum threshold
                if silence_duration >= min_silence:
                    silence_gaps.append((
                        round(silence_start, 3),
                        round(silence_end, 3),
                        round(silence_duration, 3)
                    ))
                
                silence_start = None
        
        return silence_gaps
        
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"ffmpeg command timed out for file: {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        raise RuntimeError(f"Error detecting silence: {str(e)}")


def get_audio_info(file_path: str) -> dict:
    """
    Get basic audio information from a file using ffprobe.
    
    Args:
        file_path: Path to the video/audio file
    
    Returns:
        Dictionary with audio duration and other info
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration,size,bit_rate',
        '-show_entries', 'stream=codec_name,codec_type,duration',
        '-of', 'default=noprint_wrappers=1',
        file_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        info = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value
        
        return info
        
    except Exception as e:
        raise RuntimeError(f"Error getting audio info: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_analysis.py <video_file> [min_silence_seconds]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    min_silence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    
    print(f"Detecting silence in: {video_path}")
    print(f"Minimum silence duration: {min_silence}s")
    print("-" * 50)
    
    gaps = detect_silence(video_path, min_silence=min_silence)
    
    if gaps:
        print(f"Found {len(gaps)} silence gaps:\n")
        for i, (start, end, duration) in enumerate(gaps, 1):
            print(f"  Gap {i}: {start}s -> {end}s (duration: {duration}s)")
    else:
        print("No silence gaps found.")
