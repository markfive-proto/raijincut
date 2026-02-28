#!/usr/bin/env python3
"""
Advanced Video Transitions
Using FFmpeg xfade for professional transitions
"""

import subprocess
import os
from typing import List

# Available transitions (from FFmpeg xfade)
TRANSITIONS = {
    "fade": "fade",
    "crossfade": "crossfade", 
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "wipedown": "wipedown",
    "wipeup": "wipeup",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "slidedown": "slidedown",
    "slideup": "slideup",
    "circlecrop": "circlecrop",
    "rectcrop": "rectcrop",
    "distance": "distance",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "radial": "radial",
    "smoothleft": "smoothleft",
    "smoothright": "smoothright",
    "smoothup": "smoothup",
    "smoothdown": "smoothdown",
    "pixelize": "pixelize",
    "diagtl": "diagtl",  # diagonal top-left
    "diagtr": "diagtr",  # diagonal top-right
    "diabl": "diabl",    # diagonal bottom-left
    "diabr": "diabr",    # diagonal bottom-right
    "hlslice": "hlslice",
    "hrslice": "hrslice",
    "vuslice": "vuslice",
    "vdslice": "vdslice",
    "dissolve": "dissolve",
    "patch": "patch",
    "random": "random"
}

def create_transitioned_reel(
    clip_paths: List[str],
    output_path: str,
    transition: str = "crossfade",
    transition_duration: float = 0.5
) -> str:
    """
    Create a reel with professional transitions between clips
    
    Args:
        clip_paths: List of video clip paths
        output_path: Output file path
        transition: Transition name (see TRANSITIONS dict)
        transition_duration: Duration of each transition in seconds
    
    Returns:
        Path to output file
    """
    if len(clip_paths) < 2:
        # No transition needed for single clip
        if clip_paths:
            import shutil
            shutil.copy(clip_paths[0], output_path)
        return output_path
    
    transition = transition.lower()
    if transition not in TRANSITIONS:
        print(f"   ⚠️ Unknown transition '{transition}', using 'crossfade'")
        transition = "crossfade"
    
    print(f"   Creating reel with {transition} transitions ({transition_duration}s each)")
    
    # Build FFmpeg filter complex for xfade
    # xfade works by having multiple inputs and transitioning between them
    
    if len(clip_paths) == 2:
        # Simple case: two clips
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_paths[0],
            "-i", clip_paths[1],
            "-filter_complex",
            f"[0:v][1:v]xfade=transition={TRANSITIONS[transition]}:duration={transition_duration}:offset={get_clip_duration(clip_paths[0]) - transition_duration}[v]",
            "-map", "[v]",
            "-map", "0:a",  # Keep audio from first clip
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
    else:
        # Multiple clips - more complex
        # Build xfade chain
        filter_parts = []
        
        # First clip
        duration_1 = get_clip_duration(clip_paths[0])
        
        for i in range(1, len(clip_paths)):
            duration_prev = get_clip_duration(clip_paths[i-1])
            offset = duration_prev - transition_duration
            
            if i == 1:
                filter_parts.append(f"[0:v][1:v]xfade=transition={TRANSITIONS[transition]}:duration={transition_duration}:offset={offset}[v{i}]")
            else:
                filter_parts.append(f"[v{i-1}][{i}:v]xfade=transition={TRANSITIONS[transition]}:duration={transition_duration}:offset={offset}[v{i}]")
        
        filter_complex = ";".join(filter_parts)
        
        # Build input args
        inputs = []
        for path in clip_paths:
            inputs.extend(["-i", path])
        
        cmd = [
            "ffmpeg", "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[v{len(clip_paths)-1}]",
            "-map", "0:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️ Transition failed: {result.stderr[:200]}")
        # Fallback to simple concatenation
        return create_simple_reel(clip_paths, output_path)
    
    return output_path

def create_simple_reel(clip_paths: List[str], output_path: str) -> str:
    """Fallback: simple concatenation without transitions"""
    concat_file = output_path + ".txt"
    with open(concat_file, 'w') as f:
        for path in clip_paths:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(concat_file):
        os.remove(concat_file)
    
    return output_path

def get_clip_duration(clip_path: str) -> float:
    """Get duration of a video clip"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        clip_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 10.0  # Default

def list_transitions():
    """Print available transitions"""
    print("Available transitions:")
    for name, desc in TRANSITIONS.items():
        print(f"  {name}")

if __name__ == "__main__":
    list_transitions()
