#!/usr/bin/env python3
"""
AI-Powered Filler Decision Maker
Uses OpenAI to intelligently decide which fillers to remove
"""

import os
import json
import openai
from typing import List, Dict, Tuple

# Load API key
CONFIG_PATH = os.path.expanduser("~/.openclaw/video-transcribe-config.json")
_api_key = None

def get_openai_client():
    """Get OpenAI client with API key"""
    global _api_key
    if _api_key is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(
                f"Config file not found: {CONFIG_PATH}\n"
                "Set OPENAI_API_KEY env var or create the config file."
            )
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            _api_key = config.get("openai_api_key") or config.get("api_key")
    
    if not _api_key:
        raise ValueError("No OpenAI API key found in config")
    
    return openai.OpenAI(api_key=_api_key)

def load_transcript_context(transcript_path: str, filler_timestamps: List[Tuple], context_seconds: int = 5) -> List[Dict]:
    """
    Load transcript segments around each filler for AI to analyze
    
    Returns list of:
    {
        'filler': {...},
        'context_before': "text before",
        'context_after': "text after", 
        'full_context': "text around filler"
    }
    """
    import re
    
    # Parse SRT to get all segments
    segments = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: **[00:00 - 00:00]**  text
    pattern = r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*\s*\n(.*?)(?=\n\*\*\[\d|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for start_ts, end_ts, text in matches:
        # Parse timestamp
        parts = start_ts.split(':')
        if len(parts) == 2:
            start_sec = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            start_sec = 0
        
        segments.append({
            'start': start_sec,
            'end': parse_timestamp(end_ts),
            'text': text.strip()
        })
    
    # Build context around each filler
    contexts = []
    for filler_start, filler_end, filler_type, filler_text in filler_timestamps:
        # Find segments before and after
        before = []
        after = []
        
        for seg in segments:
            if seg['end'] <= filler_start:
                before.append(seg['text'])
            elif seg['start'] >= filler_end:
                after.append(seg['text'])
                if len(after) >= 2:
                    break
        
        context_before = ' '.join(before[-2:]) if before else ''
        context_after = ' '.join(after[:2]) if after else ''
        full_context = context_before + ' [' + filler_text + '] ' + context_after
        
        contexts.append({
            'filler': {
                'start': filler_start,
                'end': filler_end,
                'type': filler_type,
                'text': filler_text
            },
            'context_before': context_before[:200],
            'context_after': context_after[:200],
            'full_context': full_context[:500]
        })
    
    return contexts

def parse_timestamp(ts: str) -> float:
    """Parse timestamp string to seconds"""
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(ts)

def ai_decide_fillers(
    transcript_path: str,
    filler_timestamps: List[Tuple],
    max_fillers: int = 20
) -> Dict:
    """
    Use AI to decide which fillers to remove
    
    Returns:
    {
        'decisions': [
            {
                'start': float,
                'end': float,
                'action': 'remove' | 'shorten' | 'keep',
                'reason': '...'
            }
        ],
        'summary': '...',
        'model': 'gpt-4o-mini'
    }
    """
    client = get_openai_client()
    
    # Get contexts for top fillers (limit to avoid API overload)
    contexts = load_transcript_context(transcript_path, filler_timestamps[:max_fillers])
    
    if not contexts:
        return {
            'decisions': [],
            'summary': 'No fillers to analyze',
            'model': 'gpt-4o-mini'
        }
    
    # Build prompt
    filler_list = []
    for i, ctx in enumerate(contexts):
        f = ctx['filler']
        filler_list.append(f"""
{i+1}. Filler: "{f['text']}" (type: {f['type']})
   Time: {format_time(f['start'])} - {format_time(f['end'])}
   Context: ...{ctx['context_before']}[FILLER]{ctx['context_after']}...
""")
    
    prompt = f"""You are a professional video editor. Analyze filler words in a transcript and decide what to remove.

Rules:
- Remove obvious fillers: "um", "uh", "er", "ah", repeated "you know"
- Shorten borderline fillers: "like", "you know", "i think", "kind of"
- Keep meaningful pauses and natural speech patterns
- Consider if removing would make the sentence grammatically incorrect
- Consider if the filler is at a natural sentence boundary

For each filler, respond with ONE line:
- REMOVE: If safe to remove without affecting meaning
- SHORTEN: If could be shortened but be careful
- KEEP: If removing would make sentence awkward

Respond in format:
[number]. {action} - {brief_reason}

Fillers to analyze:
{''.join(filler_list)}

Only respond with the numbered decisions, nothing else."""

    # Call AI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional video editor specializing in cleaning up spoken content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    # Parse response
    decisions = []
    lines = response.choices[0].message.content.strip().split('\n')
    
    for line in lines:
        # Parse: "1. REMOVE - reason"
        match = re.match(r'(\d+)\.\s*(REMOVE|SHORTEN|KEEP)\s*-\s*(.+)', line, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            action = match.group(2).lower()
            reason = match.group(3).strip()
            
            if idx < len(contexts):
                ctx = contexts[idx]
                decisions.append({
                    'start': ctx['filler']['start'],
                    'end': ctx['filler']['end'],
                    'action': action,
                    'reason': reason,
                    'original_text': ctx['filler']['text']
                })
    
    # Count
    remove_count = sum(1 for d in decisions if d['action'] == 'remove')
    shorten_count = sum(1 for d in decisions if d['action'] == 'shorten')
    keep_count = sum(1 for d in decisions if d['action'] == 'keep')
    
    return {
        'decisions': decisions,
        'summary': f"AI decided: {remove_count} remove, {shorten_count} shorten, {keep_count} keep",
        'model': 'gpt-4o-mini',
        'raw_response': response.choices[0].message.content
    }

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS"""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) > 1:
        transcript = sys.argv[1]
        # Dummy fillers for testing
        test_fillers = [
            (10.5, 11.0, "um", "um"),
            (25.0, 25.5, "you know", "you know"),
            (60.0, 60.5, "like", "like")
        ]
        
        result = ai_decide_fillers(transcript, test_fillers)
        print(result['summary'])
        print(result['raw_response'])
