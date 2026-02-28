#!/usr/bin/env python3
"""
Video Format Converter
Converts video aspect ratio for different platforms (YouTube, Reels, TikTok, Shorts)
"""

import subprocess
import os
from typing import Tuple, Optional

# Platform specifications
PLATFORMS = {
    "youtube": {
        "aspect": "16:9",
        "resolution": (1920, 1080),
        "name": "YouTube"
    },
    "reels": {
        "aspect": "9:16",
        "resolution": (1080, 1920),
        "name": "Instagram Reels"
    },
    "tiktok": {
        "aspect": "9:16",
        "resolution": (1080, 1920),
        "name": "TikTok"
    },
    "shorts": {
        "aspect": "9:16",
        "resolution": (1080, 1920),
        "name": "YouTube Shorts"
    },
    "instagram": {
        "aspect": "1:1",
        "resolution": (1080, 1080),
        "name": "Instagram Feed"
    },
    "square": {
        "aspect": "1:1",
        "resolution": (1080, 1080),
        "name": "Square"
    },
    "landscape": {
        "aspect": "16:9",
        "resolution": (1920, 1080),
        "name": "Landscape"
    },
    "portrait": {
        "aspect": "9:16",
        "resolution": (1080, 1920),
        "name": "Portrait"
    }
}

def get_video_info(video_path: str) -> dict:
    """Get video dimensions using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and result.stdout.strip():
        w, h = map(int, result.stdout.strip().split(','))
        return {"width": w, "height": h, "aspect": w/h}
    
    return {"width": 1920, "height": 1080, "aspect": 16/9}

def convert_format(
    input_path: str,
    output_path: str,
    target_format: str = "reels",
    mode: str = "pad",  # "pad", "crop", "stretch"
    blur_background: bool = True,
    blur_amount: int = 20
) -> str:
    """
    Convert video to different aspect ratio
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_format: One of youtube, reels, tiktok, shorts, instagram, square
        mode: "pad" (add bars), "crop" (cut edges), "stretch" (distort)
        blur_background: For pad mode, blur the padded areas
    
    Returns:
        Path to output video
    """
    target = PLATFORMS.get(target_format.lower(), PLATFORMS["reels"])
    target_w, target_h = target["resolution"]
    target_aspect = target_w / target_h
    
    # Get source info
    source = get_video_info(input_path)
    source_aspect = source["width"] / source["height"]
    
    print(f"   Converting: {source['width']}x{source['height']} ({source_aspect:.2f}) → {target_w}x{target_h} ({target_aspect:.2f})")
    print(f"   Mode: {mode}")
    
    if mode == "stretch":
        # Simple stretch (not recommended - distorts video)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale={target_w}:{target_h}",
            "-c:a", "copy",
            output_path
        ]
    
    elif mode == "crop":
        # Crop to fit - center crop (loses edges)
        # FFmpeg crop syntax: crop=width:height:x:y
        # Scale first to fill target dimensions, then crop excess
        
        if source_aspect > target_aspect:
            # Source is wider than target - crop left/right edges
            # Scale to fill height, then crop width
            # scale to target height, let width overflow
            vf = f"scale=-2:{target_h},crop={target_w}:{target_h}:(iw-{target_w})/2:0"
        else:
            # Source is taller than target - crop top/bottom
            # Scale to fill width, then crop height
            vf = f"scale={target_w}:-2,crop={target_w}:{target_h}:0:(ih-{target_h})/2"
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:a", "copy",
            output_path
        ]
    
    elif mode == "pad":
        # Simple pad with black bars - more reliable
        if abs(source_aspect - target_aspect) < 0.1:
            # Already close to target aspect, just scale
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"scale={target_w}:{target_h}",
                "-c:a", "copy",
                output_path
            ]
        else:
            # Scale to fill one dimension, pad the other
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
                "-c:a", "copy",
                output_path
            ]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️ Error: {result.stderr[:200]}")
        # Fallback to simple copy
        import shutil
        shutil.copy(input_path, output_path)
    
    return output_path

def smart_crop(
    input_path: str,
    output_path: str,
    target_format: str = "reels",
    face_detect: bool = False
) -> str:
    """
    Smart crop - attempts to keep subject in frame
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_format: Target platform format
        face_detect: If True, use face detection to find best crop region
    
    Returns:
        Path to output video
    """
    if face_detect:
        try:
            import cv2
            cap = cv2.VideoCapture(input_path)
            
            # Try to find face in first few frames
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            face_x, face_y, face_w, face_h = None, None, None, None
            frame_count = 0
            
            while frame_count < 30:  # Check first 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Use the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    face_x, face_y, face_w, face_h = face
                    break
                frame_count += 1
            
            cap.release()
            
            if face_x is not None:
                # Calculate center of face
                frame_h, frame_w = frame.shape[:2]
                face_center_x = face_x + face_w // 2
                face_center_y = face_y + face_h // 2
                
                target = PLATFORMS.get(target_format.lower(), PLATFORMS["reels"])
                target_w, target_h = target["resolution"]
                
                # Calculate crop position centered on face
                source = get_video_info(input_path)
                
                # Scale to target height
                scale = target_h / source['height']
                scaled_w = source['width'] * scale
                
                # Face position in scaled frame
                scaled_face_x = face_center_x * scale
                
                # Calculate crop x offset (centered on face, clamped to bounds)
                crop_x = int(scaled_face_x - target_w / 2)
                crop_x = max(0, min(crop_x, int(scaled_w - target_w)))
                
                vf = f"scale=-2:{target_h},crop={target_w}:{target_h}:{crop_x}:0"
                
                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-vf", vf,
                    "-c:a", "copy",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   🎯 Face-detected crop: face at {face_center_x},{face_center_y}")
                    return output_path
                    
        except ImportError:
            print(f"   ⚠️ OpenCV not available, using center crop")
        except Exception as e:
            print(f"   ⚠️ Face detection failed: {e}, using center crop")
    
    # Fallback to center crop
    return convert_format(input_path, output_path, target_format, mode="crop")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        format_type = sys.argv[2]
        
        output_file = input_file.replace(".mp4", f"_{format_type}.mp4")
        
        result = convert_format(input_file, output_file, format_type)
        print(f"Converted: {result}")
    else:
        print("Available formats:")
        for k, v in PLATFORMS.items():
            print(f"  {k}: {v['name']} ({v['resolution'][0]}x{v['resolution'][1]})")
