#!/usr/bin/env python3
"""
AI-Powered Highlight Selector
Keyword pre-scoring + GPT-4o-mini refinement to pick 3-5 highlight segments
for a 60-90 second reel.

Usage:
    python highlight_selector.py transcript.json -o highlights.json
    python highlight_selector.py transcript.json -o highlights.json --max-duration 90
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

# ── Reuse scoring logic from video_repurpose.py ──

VIRAL_KEYWORDS = {
    "controversy": 2.5, "disagree": 2.2, "uncomfortable": 2.0,
    "most relevant": 2.8, "important": 1.8, "critical": 1.8,
    "dangerous": 2.5, "risk": 1.8, "fear": 1.8, "worried": 1.8,
    "will": 1.5, "going to": 1.5, "predict": 2.2, "future": 2.0,
    "expect": 1.8, "probably": 1.3, "maybe": 1.2,
    "advice": 3.0, "should": 2.0, "recommend": 2.5,
    "tip": 2.5, "learn": 1.8, "how to": 2.0, "recommendation": 2.5,
    "my advice": 3.0, "i recommend": 3.0, "i would say": 2.0,
    "billion": 2.8, "million": 2.5, "percent": 2.5,
    "10x": 3.0, "100x": 3.0, "10,000": 2.2,
    "AI": 1.5, "artificial intelligence": 1.8, "model": 1.5,
    "AGI": 2.5, "artificial general intelligence": 2.8,
    "career": 2.5, "job": 2.2, "jobs": 2.2, "employment": 2.2,
    "replace": 2.0, "disrupt": 2.2, "disruption": 2.2,
    "opportunity": 2.0, "startup": 1.8, "entrepreneur": 1.8,
    "india": 2.0, "indian": 2.0, "country": 1.8, "economy": 2.2, "economic": 2.0,
    "biotech": 2.8, "biology": 2.5, "health": 2.0, "medicine": 2.2,
    "cancer": 2.8, "cure": 2.8, "disease": 2.5,
    "regulation": 2.2, "government": 1.8, "policy": 1.8, "law": 1.8,
    "open source": 2.5, "google": 1.5, "openai": 1.5, "anthropic": 1.5,
    "deepseek": 2.0, "china": 1.8, "american": 1.5,
    "consciousness": 2.5, "sentient": 2.2, "think": 1.5, "believe": 1.3,
    "power": 2.0, "control": 1.8, "concentration": 2.0, "moat": 2.8,
    "competitive": 2.0, "advantage": 2.0,
    "human": 1.5, "society": 2.0, "world": 1.5, "people": 1.3,
    "stupider": 2.5, "dumber": 2.2,
}

ANSWER_INDICATORS = [
    r"^I think", r"^I believe", r"^I would say", r"^I mean",
    r"^My view", r"^I don't think", r"^I agree", r"^I disagree",
    r"^The key", r"^One is", r"^I want to", r"^Let me",
    r"^Here's", r"^So,", r"^You know,",
]

QUESTION_PATTERNS = [
    r"^\w+\s+do\s+you", r"^\w+\s+what\s+is", r"^\w+\s+how\s+do",
    r"^\w+\s+can\s+you", r"^\w+\s+will\s+you", r"^\w+\s+would\s+you",
    r"^\w+\s+do\s+you\s+think", r"^\w+\s+are\s+you",
]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    score: float = 0.0
    reason: str = ""


def score_segment(seg: Segment) -> float:
    text_lower = seg.text.lower()
    score = 0.0
    reasons = []

    for keyword, weight in VIRAL_KEYWORDS.items():
        count = text_lower.count(keyword)
        if count > 0:
            score += weight * min(count, 3)
            if count == 1:
                reasons.append(keyword)
            else:
                reasons.append(f"{keyword}x{count}")

    for pattern in ANSWER_INDICATORS:
        if re.search(pattern, seg.text, re.IGNORECASE):
            score += 1.5
            reasons.append("answer")
            break

    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, seg.text, re.IGNORECASE):
            score -= 1.5
            break

    duration = seg.end - seg.start
    if 30 <= duration <= 60:
        score += 2.0
    elif duration > 60:
        score += 1.0
    elif duration >= 15:
        score += 0.5
    elif duration >= 5:
        score += 0.2

    sentences = len([s for s in re.split(r'[.!?]+', seg.text) if s.strip()])
    if 2 <= sentences <= 6:
        score += 1.2

    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', seg.text))
    if 1 <= caps_words <= 3:
        score += 0.8

    seg.score = score
    seg.reason = ", ".join(reasons[:5])
    return score


# ── OpenAI client (same pattern as ai_filler_decider.py) ──

CONFIG_PATH = os.path.expanduser("~/.openclaw/video-transcribe-config.json")


def get_openai_client():
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                config = json.load(f)
                api_key = config.get("openai_api_key") or config.get("api_key")
    if not api_key:
        raise ValueError("No OpenAI API key found (env OPENAI_API_KEY or config file)")
    return openai.OpenAI(api_key=api_key)


# ── Core logic ──

def load_segments(transcript_path: str) -> List[Segment]:
    with open(transcript_path) as f:
        data = json.load(f)

    segments = []
    for seg in data.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=text,
        ))
    return segments


def merge_short_segments(segments: List[Segment], min_duration: float = 10.0) -> List[Segment]:
    """Merge adjacent short segments into longer ones for better scoring."""
    if not segments:
        return []

    merged = [Segment(start=segments[0].start, end=segments[0].end, text=segments[0].text)]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start - prev.end
        prev_dur = prev.end - prev.start
        if gap < 2.0 and prev_dur < min_duration:
            prev.end = seg.end
            prev.text = prev.text + " " + seg.text
        else:
            merged.append(Segment(start=seg.start, end=seg.end, text=seg.text))
    return merged


def keyword_select(segments: List[Segment], max_duration: float = 90.0, top_n: int = 5) -> List[Segment]:
    """Pure keyword-based fallback selection."""
    for seg in segments:
        score_segment(seg)

    ranked = sorted(segments, key=lambda s: s.score, reverse=True)

    selected = []
    total_dur = 0.0
    for seg in ranked:
        dur = seg.end - seg.start
        if total_dur + dur > max_duration:
            continue
        selected.append(seg)
        total_dur += dur
        if len(selected) >= top_n:
            break

    selected.sort(key=lambda s: s.start)
    return selected


def ai_select(segments: List[Segment], max_duration: float = 90.0) -> Optional[List[Segment]]:
    """Use GPT-4o-mini to select highlights from pre-scored candidates."""
    for seg in segments:
        score_segment(seg)

    ranked = sorted(segments, key=lambda s: s.score, reverse=True)
    candidates = ranked[:30]

    candidate_text = []
    for i, seg in enumerate(candidates):
        dur = seg.end - seg.start
        candidate_text.append(
            f"{i+1}. [{seg.start:.1f}s - {seg.end:.1f}s] (score={seg.score:.1f}, {dur:.0f}s)\n"
            f"   \"{seg.text[:200]}\""
        )

    newline = "\n"
    prompt = f"""You are a professional video editor creating a 60-90 second highlight reel.
Given these transcript segments with scores, select 3-5 segments that:
- Tell a coherent story or cover the most impactful points
- Are self-contained (make sense without surrounding context)
- Prioritize answers/insights over questions
- Total duration: 60-{int(max_duration)} seconds

Segments:
{newline.join(candidate_text)}

Return ONLY valid JSON array, no other text:
[{{"index": 1, "reason": "why this is compelling"}}]
The "index" is the segment number from the list above (1-based)."""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional video editor. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        picks = json.loads(raw)
        selected = []
        total_dur = 0.0
        for pick in picks:
            idx = int(pick["index"]) - 1
            if 0 <= idx < len(candidates):
                seg = candidates[idx]
                seg.reason = pick.get("reason", seg.reason)
                dur = seg.end - seg.start
                if total_dur + dur <= max_duration + 5:
                    selected.append(seg)
                    total_dur += dur

        if selected:
            selected.sort(key=lambda s: s.start)
            return selected

    except Exception as e:
        print(f"AI selection failed ({e}), falling back to keyword scoring", file=sys.stderr)

    return None


def select_highlights(transcript_path: str, output_path: str, max_duration: float = 90.0):
    segments = load_segments(transcript_path)
    if not segments:
        print("No segments found in transcript", file=sys.stderr)
        with open(output_path, "w") as f:
            json.dump([], f)
        return

    merged = merge_short_segments(segments)
    print(f"Loaded {len(segments)} segments, merged into {len(merged)} candidate groups")

    # Try AI selection first, fall back to keyword-only
    selected = ai_select(merged, max_duration)
    method = "ai"
    if selected is None:
        selected = keyword_select(merged, max_duration)
        method = "keyword"

    total_dur = sum(s.end - s.start for s in selected)
    print(f"Selected {len(selected)} highlights ({total_dur:.1f}s total) via {method}")
    for i, seg in enumerate(selected, 1):
        dur = seg.end - seg.start
        print(f"  {i}. [{seg.start:.1f}s - {seg.end:.1f}s] ({dur:.0f}s) score={seg.score:.1f} — {seg.reason}")

    highlights = [asdict(seg) for seg in selected]
    with open(output_path, "w") as f:
        json.dump(highlights, f, indent=2)
    print(f"Saved highlights to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered highlight selector")
    parser.add_argument("transcript", help="Path to transcript.json from transcribe_only.py")
    parser.add_argument("-o", "--output", required=True, help="Output highlights.json path")
    parser.add_argument("--max-duration", type=float, default=90.0, help="Max total highlight duration in seconds")
    args = parser.parse_args()

    select_highlights(args.transcript, args.output, args.max_duration)
