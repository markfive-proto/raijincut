"""Self-check for analyze.py's pure logic. Run: python3 python/test_analyze.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from analyze import build_shots, detect_boundaries, group_words, onsets, pacing

fps = 30
frames = [{"t": i / fps, "score": 0.005, "y": 120.0} for i in range(300)]  # 10 s
frames[60]["score"] = 0.8                      # hard cut at 2.0 s
for i in range(150, 160):                      # 5.0-5.3 s: luma ramps into black, then back
    frames[i]["y"] = 120 - (i - 150) * 10
for i in range(160, 166):
    frames[i]["y"] = 16.0
# gray thumbnails: shot A (value 40) crossfades into shot B (value 200) over 7.2-7.8 s
g = [bytes([40] * 16)] * 216 + [bytes([40 + round(160 * k / 18)] * 16) for k in range(18)] + [bytes([200] * 16)] * 66

b = detect_boundaries(frames, fps, g=g)
types = [(round(x["t"], 1), x["type"]) for x in b]
assert types == [(2.0, "cut"), (5.4, "fade_black"), (7.5, "dissolve")], types

shots = build_shots(b, 10.0)
assert [s["transition_in"]["detected"] for s in shots] == ["start", "cut", "fade_black", "dissolve"]
p = pacing(shots, 10.0)
assert p["cut_count"] == 3 and p["cuts_per_10s"] == 3.0 and p["cuts_in_first_3s"] == 1, p

words = [{"text": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate("one two three. four five six seven eight nine ten".split())]
cues = group_words(words)
assert [c["text"] for c in cues] == ["one two three.", "four five six seven eight nine ten"], cues

env = [-60.0] * 100
env[50] = -20.0                                # one sharp hit
assert [i for i, _, _ in onsets(env)] == [50]
print("ok")
