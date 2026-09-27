"""Self-checks for every command on ffmpeg lavfi samples. No network, no whisper, no vision.
Run: python3 tests/test_raijincut.py   (or python -m pytest tests)"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raijincut.analyze import build_shots, detect_boundaries, group_words, onsets, pacing, parse_srt
from raijincut.cli import download_error
from raijincut.roughcut import plan_edits

TMP = tempfile.mkdtemp(prefix="raijincut-test-")


def sh(*args):
    subprocess.run(args, check=True, capture_output=True)


def rc(*args):
    """Run the CLI; return stdout."""
    r = subprocess.run([sys.executable, "-m", "raijincut", *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def dur(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)


def size(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                   "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
    return j["width"], j["height"]


def sample(name="talk.mp4", seconds=8, s="640x360"):
    p = os.path.join(TMP, name)
    if not os.path.exists(p):
        sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={s}:d={seconds}", "-f", "lavfi",
           "-i", f"sine=f=220:d={seconds}", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p)
    return p


def test_analyze_logic():
    fps = 30
    frames = [{"t": i / fps, "score": 0.005, "y": 120.0} for i in range(300)]  # 10 s
    frames[60]["score"] = 0.8                      # hard cut at 2.0 s
    for i in range(150, 160):                      # 5.0-5.3 s: luma ramps into black, then back
        frames[i]["y"] = 120 - (i - 150) * 10
    for i in range(160, 166):
        frames[i]["y"] = 16.0
    # gray thumbnails: shot A (40) crossfades into shot B (200) over 7.2-7.8 s; fed as a stream
    g = [bytes([40] * 16)] * 216 + [bytes([40 + round(160 * k / 18)] * 16) for k in range(18)] + [bytes([200] * 16)] * 66
    b = detect_boundaries(frames, fps, g=iter(g))
    assert [(round(x["t"], 1), x["type"]) for x in b] == [(2.0, "cut"), (5.4, "fade_black"), (7.5, "dissolve")], b
    shots = build_shots(b, 10.0)
    assert [s["transition_in"]["detected"] for s in shots] == ["start", "cut", "fade_black", "dissolve"]
    p = pacing(shots, 10.0)
    assert p["cut_count"] == 3 and p["cuts_per_10s"] == 3.0 and p["cuts_in_first_3s"] == 1, p
    words = [{"text": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate("one two three. four five six seven eight nine ten".split())]
    assert [c["text"] for c in group_words(words)] == ["one two three.", "four five six seven eight nine ten"]
    env = [-60.0] * 100
    env[50] = -20.0
    assert [i for i, _, _ in onsets(env)] == [50]


WORDS = [("So", 0.2, 0.5), ("um", 0.6, 0.9), ("we", 1.0, 1.2), ("we", 1.3, 1.5), ("build", 1.6, 2.0),
         ("apps.", 5.0, 5.5), ("嗯", 5.6, 5.9), ("fast.", 6.2, 6.6)]


def test_plan_edits():
    keep, edits = plan_edits([{"w": w, "start": a, "end": b} for w, a, b in WORDS], 0.6, 8.0)
    assert sorted(e["reason"] for e in edits) == ["filler", "filler", "pause", "repeat"], edits
    assert len(keep) == 4 and keep[0]["start"] == 0.05 and keep[-1]["end"] == 6.9, keep


def test_rough_cut():
    src = sample()
    wj = os.path.join(TMP, "words.json")
    json.dump({"words": [{"w": w, "start": a, "end": b} for w, a, b in WORDS]}, open(wj, "w"))
    out = os.path.join(TMP, "rough.mov")
    rc("rough-cut", src, "--words", wj, "-o", out)
    r = json.load(open(os.path.join(TMP, "rough.edits.json")))
    kept = sum(k["end"] - k["start"] for k in r["keep"])
    assert abs(dur(out) - kept) < 0.25 and dur(out) < 8 - 3, (dur(out), kept)
    assert [w["w"] for w in r["words"]] == ["So", "we", "build", "apps.", "fast."]
    assert parse_srt(os.path.join(TMP, "rough.srt"))[0][2].startswith("So we build")


def test_probe():
    out = rc("probe", sample())
    assert "Video: 640x360" in out and "Audio: aac" in out, out


def test_cut():
    out = os.path.join(TMP, "cut.mp4")
    rc("cut", sample(), "-s", "1", "-e", "3.5", "-o", out)
    assert abs(dur(out) - 2.5) < 0.1, dur(out)


def test_crop():
    out = os.path.join(TMP, "crop.mp4")
    rc("crop", sample(), "-a", "9:16", "-o", out)
    assert size(out) == (202, 360), size(out)


def test_convert():
    out = os.path.join(TMP, "conv.mp4")
    rc("convert", sample(), "-o", out, "-c", "30")
    assert abs(dur(out) - 8) < 0.1


def gray(path, t):
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-vf", "scale=64:36,format=gray",
                           "-f", "rawvideo", "-"], capture_output=True).stdout


def test_subtitle():
    srt = os.path.join(TMP, "s.srt")
    open(srt, "w").write("1\n00:00:01,000 --> 00:00:02,000\nHELLO CAPTION\n\n2\n00:00:04,000 --> 00:00:05,000\nsecond line here\n")
    out = os.path.join(TMP, "subbed.mp4")
    rc("subtitle", sample(), "--srt", srt, "-o", out)
    assert abs(dur(out) - 8) < 0.1
    assert gray(out, 1.5) != gray(sample(), 1.5), "caption burned in its window"
    assert sum(abs(a - b) for a, b in zip(gray(out, 3.0), gray(sample(), 3.0))) < 64 * 36 * 3, "no caption between cues"


def test_repurpose():
    out = os.path.join(TMP, "short.mp4")
    rc("repurpose", sample(), "-p", "shorts", "-c", "00:00:01..00:00:03", "-o", out)
    assert size(out) == (1080, 1920) and abs(dur(out) - 2) < 0.1, (size(out), dur(out))


def test_presets():
    assert "shorts          1080x1920" in rc("presets")


def test_download_hint():
    assert "--cookies-from-browser chrome" in download_error("https://www.linkedin.com/posts/x", "Unable to extract", None)
    assert download_error("https://youtube.com/watch?v=1", "HTTP Error 404", None).startswith("yt-dlp failed")


def test_analyze():
    # no audio stream, so transcription and audio analysis are skipped: shots, keyframes, breakdown only
    src = os.path.join(TMP, "cuts.mp4")
    sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x568:d=2:r=30", "-f", "lavfi",
       "-i", "color=c=blue:s=320x568:d=2:r=30", "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", src)
    rc("analyze", src, "-o", TMP, "--vision", "none")
    b = json.load(open(os.path.join(TMP, "cuts", "breakdown.json")))
    assert [(round(x["t"], 1), x["type"]) for x in b["shots"]["boundaries"]] == [(2.0, "cut")], b["shots"]["boundaries"]
    assert b["pacing"]["shot_count"] == 2 and os.path.exists(os.path.join(TMP, "cuts", "breakdown.md"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all ok")
