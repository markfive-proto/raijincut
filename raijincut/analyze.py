"""Break a video down into transcript, shots, keyframes, vision notes, audio hits
and captions, then write breakdown.json + breakdown.md. Entry point: `raijincut analyze`.

Memory: frames and audio are streamed from ffmpeg pipes one small chunk at a time (32 px gray
frames, 10 ms audio hops). Nothing holds the whole video or full-res frames in memory.
Pillow draws strips and the contact sheet. The `anthropic` SDK is optional (only for --vision api).
"""
import base64, datetime, json, math, os, re, shutil, statistics, subprocess, sys, tempfile, urllib.request
from array import array
from collections import deque
from itertools import islice

SCHEMA_VERSION = 1
SR = 16000          # analysis sample rate
HOP = 160           # 10 ms envelope hop


def log(msg):
    print(msg, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def slugify(s, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n].strip("-") or "video"


def ts(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


# ---------------------------------------------------------------- probe

def probe(video):
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video])
    if out.returncode:
        sys.exit(f"ffprobe failed: {out.stderr.strip()}")
    j = json.loads(out.stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    if not v:
        sys.exit("No video stream found.")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "duration_s": round(float(j["format"].get("duration", 0)), 3),
        "width": v.get("width"), "height": v.get("height"), "fps": round(fps, 3),
        "has_audio": any(s["codec_type"] == "audio" for s in j["streams"]),
    }


# ---------------------------------------------------------------- transcript

def group_words(words, max_words=7, max_dur=3.0):
    """Group word dicts into caption cues: break on sentence end, word count or duration."""
    cues, cur = [], []
    for w in words:
        cur.append(w)
        if (len(cur) >= max_words or w["end"] - cur[0]["start"] >= max_dur
                or w["text"].rstrip().endswith((".", "?", "!"))):
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    return [{"start": c[0]["start"], "end": c[-1]["end"], "text": " ".join(x["text"] for x in c)} for c in cues]


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues, path):
    with open(path, "w") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}\n\n")


def parse_srt(path):
    """[(start, end, text)] from an SRT file."""
    def secs(t):
        h, m, rest = t.strip().replace(".", ",").split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    cues = []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8").read().replace("\r", "").strip()):
        lines = block.split("\n")
        if len(lines) >= 3 and "-->" in lines[1]:
            a, b = lines[1].split("-->")
            cues.append((secs(a), secs(b), "\n".join(lines[2:])))
    return cues


def transcribe(wav, whisper_model, tmp):
    """Returns (backend, language, words). Prefers whisper.cpp, then mlx_whisper."""
    if whisper_model and shutil.which("whisper-cli"):
        if not os.path.exists(whisper_model):
            sys.exit(f"Whisper model not found: {whisper_model}")
        base = os.path.join(tmp, "whisper")
        out = run(["whisper-cli", "-m", whisper_model, "-f", wav, "-ml", "1", "-sow", "-oj", "-of", base,
                   "-t", "4", "-np"])
        if out.returncode:
            raise RuntimeError(out.stderr[-800:])
        j = json.load(open(base + ".json"))
        words = []
        for seg in j.get("transcription", []):
            text = seg["text"].strip()
            if not text or text.startswith(("[", "(")):
                continue
            words.append({"text": text, "start": seg["offsets"]["from"] / 1000, "end": seg["offsets"]["to"] / 1000})
        return "whisper.cpp:" + os.path.basename(whisper_model), j.get("result", {}).get("language"), words
    if shutil.which("mlx_whisper"):
        cmd = ["mlx_whisper", wav, "--word-timestamps", "True", "-f", "json", "-o", tmp]
        model = os.environ.get("RAIJINCUT_MLX_MODEL")
        if model:
            cmd += ["--model", model]
        out = run(cmd)
        if out.returncode:
            raise RuntimeError(out.stderr[-800:])
        j = json.load(open(os.path.join(tmp, os.path.splitext(os.path.basename(wav))[0] + ".json")))
        words = [{"text": w["word"].strip(), "start": w["start"], "end": w["end"]}
                 for s in j.get("segments", []) for w in s.get("words", []) if w["word"].strip()]
        return "mlx_whisper:" + (model or "default"), j.get("language"), words
    return None, None, []


def transcribe_video(video, whisper_model, tmp):
    """16 kHz mono wav from any media file, then transcribe(). Returns (backend, language, words)."""
    wav = os.path.join(tmp, "audio.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), wav])
    return transcribe(wav, whisper_model, tmp)


# ---------------------------------------------------------------- shots

def frame_stats(video, tmp):
    """One ffmpeg pass at 160px: per-frame scene score and average luma."""
    mf = os.path.join(tmp, "frames.txt")
    vf = f"scale=160:-2,signalstats,select='gte(scene,0)',metadata=mode=print:file={mf}"
    out = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-an", "-vf", vf, "-f", "null", "-"])
    if out.returncode:
        raise RuntimeError(out.stderr[-800:])
    frames, cur = [], None
    for line in open(mf):
        if line.startswith("frame:"):
            cur = {"t": float(line.split("pts_time:")[1]), "score": 0.0, "y": 128.0}
            frames.append(cur)
        elif cur is not None and "=" in line:
            k, v = line.strip().split("=", 1)
            if k == "lavfi.scene_score":
                cur["score"] = float(v)
            elif k == "lavfi.signalstats.YAVG":
                cur["y"] = float(v)
    return frames


def stream(cmd, chunk):
    """Yield fixed-size chunks from a command's stdout, one at a time."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while len(buf := p.stdout.read(chunk)) == chunk:
            yield buf
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def thumbs(video, size=32):
    """Every frame as size x size gray bytes (1 KB), streamed one frame at a time."""
    return stream(["ffmpeg", "-v", "error", "-i", video, "-an", "-vf", f"scale={size}:{size},format=gray",
                   "-f", "rawvideo", "-"], size * size)


def mad(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def find_dissolves(frames, fps, taken, step=2, spans=()):
    """A crossfade's middle frame is the average of the frames either side of it; fast motion is not.
    ffmpeg's scene score stays near 0 during a steady crossfade, so this looks at pixels instead.
    `frames` is any iterable of gray thumbnails; only a 2*K+1 frame window is kept (K = 0.5 s).
    Returns [(t, d_ab)] for windows of 0.3-1.0 s that contain no hard cut."""
    ks = sorted({max(2, round(fps * h)) for h in (0.15, 0.3, 0.5)})
    win = deque(maxlen=2 * ks[-1] + 1)   # frames n-2K .. n
    steps = deque(maxlen=2 * ks[-1])     # mad(frame i, frame i+1) over the same window
    found = []
    for n, f in enumerate(frames):
        if win:
            steps.append(mad(win[-1], f))
        win.append(f)
        for ki, k in enumerate(ks):
            c = n - k                     # centre of a window that ends at this frame
            if c < k or (c - k) % step:
                continue
            t = c / fps
            if any(abs(t - x) <= k / fps + 0.1 for x in taken) or any(a - k / fps <= t <= b + k / fps for a, b in spans):
                continue
            a, m, b = win[-1 - 2 * k], win[-1 - k], f
            d_ab = mad(a, b)
            if d_ab < 20:
                continue
            biggest_step = max(islice(reversed(steps), 2 * k))
            err = sum(abs(z - (x + y) / 2) for x, y, z in zip(a, b, m)) / len(a)
            if err < 0.2 * d_ab and biggest_step < 0.35 * d_ab:
                found.append((ki, c, t, d_ab))
    found.sort()                          # by window size, then time: stable tie order for picking
    picked = []
    for _, _, t, d in sorted(found, key=lambda f: -f[3]):
        if all(abs(t - p) > 0.5 for p, _ in picked):
            picked.append((t, d))
    return sorted(picked)


def detect_boundaries(frames, fps, thr=0.3, min_shot=0.25, g=None):
    """Classify shot boundaries from per-frame scene scores and luma.

    - cut: a single-frame scene-score spike >= thr.
    - fade_black / fade_white: a run of near-black/near-white frames; boundary at its middle.
      Called a fade when luma ramps over >= 3 frames into the run, else cut_to_black/flash.
    - dissolve: pixel blend test on gray thumbnails `g`, any iterable (see find_dissolves).
    ponytail: fixed heuristic thresholds; the vision pass double-checks every transition.
    """
    b, spans = [], []   # spans: fade ramps + runs; a fade through black is a blend too, so dissolves skip them
    s = [f["score"] for f in frames]
    y = [f["y"] for f in frames]
    n = len(frames)

    def add(t, typ, conf, score):
        if all(abs(t - x["t"]) >= min_shot for x in b):
            b.append({"t": round(t, 3), "type": typ, "confidence": conf, "score": round(score, 3)})

    # luma runs (fades and flashes) first, they take priority over the spikes they cause
    i = 0
    while i < n:
        kind = "black" if y[i] <= 24 else "white" if y[i] >= 225 else None
        if not kind:
            i += 1
            continue
        j = i
        while j + 1 < n and ((y[j + 1] <= 24) if kind == "black" else (y[j + 1] >= 225)):
            j += 1
        if 0 < i and j < n - 1:  # ignore black at the very start or end of the video
            ramp = 0
            k = i - 1
            while k > 0 and ramp < 15 and ((y[k - 1] > y[k] + 1) if kind == "black" else (y[k - 1] < y[k] - 1)):
                ramp += 1
                k -= 1
            up = j + 1   # the ramp back out of the run
            while up + 1 < n and up - j < 15 and ((y[up + 1] > y[up] + 1) if kind == "black" else (y[up + 1] < y[up] - 1)):
                up += 1
            spans.append((frames[k]["t"], frames[up]["t"]))
            mid = (frames[i]["t"] + frames[j]["t"]) / 2
            if kind == "black":
                add(mid, "fade_black" if ramp >= 3 else "cut_to_black", 0.8, max(s[i:j + 2]))
            else:
                add(mid, "flash_white", 0.7, max(s[i:j + 2]))
        i = j + 1

    for i in range(1, n):
        if s[i] >= thr and s[i] >= s[i - 1] and (i + 1 >= n or s[i] >= s[i + 1]):
            add(frames[i]["t"], "cut", round(min(1.0, 0.5 + s[i]), 2), s[i])

    if g is not None:
        for t, d in find_dissolves(g, fps, [x["t"] for x in b], spans=spans):
            add(t, "dissolve", 0.6, d / 255)
    return sorted(b, key=lambda x: x["t"])


def build_shots(bounds, duration):
    edges = [0.0] + [x["t"] for x in bounds] + [duration]
    shots = []
    for i in range(len(edges) - 1):
        st, en = edges[i], edges[i + 1]
        tin = bounds[i - 1] if i > 0 else None
        shots.append({"index": i + 1, "start": round(st, 3), "end": round(en, 3), "duration": round(en - st, 3),
                      "transition_in": {"detected": tin["type"] if tin else "start",
                                        "confidence": tin["confidence"] if tin else 1.0}})
    return shots


def pacing(shots, duration):
    d = [x["duration"] for x in shots]
    cuts = len(shots) - 1
    longest = max(shots, key=lambda x: x["duration"])
    return {
        "shot_count": len(shots), "cut_count": cuts,
        "avg_shot_s": round(sum(d) / len(d), 2), "median_shot_s": round(statistics.median(d), 2),
        "min_shot_s": round(min(d), 2), "max_shot_s": round(max(d), 2),
        "cuts_per_10s": round(cuts / duration * 10, 2) if duration else 0,
        "cuts_in_first_3s": sum(1 for x in shots[1:] if x["start"] < 3.0),
        "longest_shot": {"index": longest["index"], "start": longest["start"], "duration": longest["duration"]},
    }


# ---------------------------------------------------------------- keyframes

def grab(video, t, path, width=480):
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", video, "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "3", path])
    return os.path.exists(path)


def keyframes(video, shots, out):
    kd = os.path.join(out, "keyframes")
    os.makedirs(kd, exist_ok=True)
    for s in shots:
        pad = min(0.08, s["duration"] / 4)
        kf = {}
        for name, t in (("first", s["start"] + pad), ("mid", (s["start"] + s["end"]) / 2), ("last", s["end"] - pad)):
            p = os.path.join(kd, f"shot{s['index']:03d}_{name}.jpg")
            if grab(video, t, p):
                kf[name] = os.path.relpath(p, out)
        s["keyframes"] = kf


def pil():
    try:
        from PIL import Image, ImageDraw
        return Image, ImageDraw
    except ImportError:
        return None, None


def contact_sheet(shots, out, cols=6, w=180):
    Image, ImageDraw = pil()
    if not Image:
        return None
    thumbs = []
    for s in shots:
        p = s.get("keyframes", {}).get("mid")
        if p:
            im = Image.open(os.path.join(out, p)).convert("RGB")
            im = im.resize((w, int(im.height * w / im.width)))
            thumbs.append((s, im))
    if not thumbs:
        return None
    h = max(im.height for _, im in thumbs) + 18
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    d = ImageDraw.Draw(sheet)
    for i, (s, im) in enumerate(thumbs):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y + 18))
        d.text((x + 3, y + 3), f"#{s['index']} {ts(s['start'])} ({s['duration']:.1f}s)", fill="white")
    path = os.path.join(out, "contact_sheet.jpg")
    sheet.save(path, quality=85)
    return "contact_sheet.jpg"


def strips(shots, out, h=320):
    """Per shot: [prev last | first | mid | last] side by side, so one image shows the transition in."""
    Image, ImageDraw = pil()
    if not Image:
        return
    sd = os.path.join(out, "strips")
    os.makedirs(sd, exist_ok=True)
    prev = None
    for s in shots:
        panels = [("prev last", prev)] + [(k, s["keyframes"].get(k)) for k in ("first", "mid", "last")]
        ims = []
        for label, p in panels:
            if p:
                im = Image.open(os.path.join(out, p)).convert("RGB")
                im = im.resize((int(im.width * h / im.height), h))
            else:
                im = Image.new("RGB", (int(h * 9 / 16), h), "gray")
            ims.append((label, im))
        strip = Image.new("RGB", (sum(im.width for _, im in ims) + 6 * len(ims), h + 16), "white")
        d = ImageDraw.Draw(strip)
        x = 0
        for label, im in ims:
            strip.paste(im, (x, 16))
            d.text((x + 3, 2), label, fill="black")
            x += im.width + 6
        path = os.path.join(sd, f"shot{s['index']:03d}.jpg")
        strip.save(path, quality=80)
        s["strip"] = os.path.relpath(path, out)
        prev = s["keyframes"].get("last")


# ---------------------------------------------------------------- audio

def envelope_db(video, af=None):
    """10 ms RMS envelope in dBFS, streamed from ffmpeg one hop at a time."""
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR)]
    if af:
        cmd += ["-af", af]
    env = []
    for buf in stream(cmd + ["-f", "s16le", "-"], HOP * 2):
        seg = array("h", buf)
        rms = math.sqrt(sum(x * x for x in seg) / HOP)
        env.append(20 * math.log10(rms / 32768 + 1e-9))
    return env


def onsets(env, rise_db=10.0, above_median_db=0.0, gap=15):
    """Local maxima of a sudden level rise vs 30-120 ms earlier. Returns [(frame, rise, level)]."""
    med = statistics.median(env) if env else -90
    cand = []
    for i in range(12, len(env)):
        rise = env[i] - sum(env[i - 12:i - 3]) / 9
        if rise >= rise_db and env[i] >= med + above_median_db:
            cand.append((i, rise, env[i]))
    picked = []
    for c in sorted(cand, key=lambda c: -c[1]):
        if all(abs(c[0] - p[0]) >= gap for p in picked):
            picked.append(c)
    return sorted(picked)


def loudness(video):
    out = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    summary = out.stderr.split("Summary:")[-1]

    def grab_num(key):
        m = re.search(rf"{key}:\s+(-?[\d.]+|-inf)", summary)
        return float(m.group(1)) if m and m.group(1) != "-inf" else None
    return {"integrated_lufs": grab_num("I"), "lra_lu": grab_num("LRA"), "true_peak_dbfs": grab_num("Peak")}


SFX_BANDS = (  # band, ffmpeg filter, min rise in dB, label
    ("low", "lowpass=f=150", 15, "impact"),          # booms, bass drops, kicks
    ("high", "highpass=f=3000", 25, "whoosh_or_click"),  # whooshes, swipes, pops, clicks
    ("full", None, 25, "hit"),
)


def audio_analysis(video, bounds):
    """Loudness, music-bed guess and SFX hit candidates aligned to cuts.
    ponytail: level-based heuristics, no source separation. Speech plosives and sibilants can pass as
    hits, so trust `aligned_to_cut` hits most. Add a music/SFX classifier if these misfire."""
    full = envelope_db(video)
    if not full:
        return {"status": "no_audio"}
    hits = []
    for band, af, rise_db, label in SFX_BANDS:
        env = full if af is None else envelope_db(video, af)
        for i, rise, lvl in onsets(env, rise_db=rise_db):
            t = i * HOP / SR
            if any(abs(t - h["t"]) < 0.08 for h in hits):
                continue
            near = min((abs(t - b["t"]) for b in bounds), default=None)
            hits.append({"t": round(t, 2), "band": band, "likely": label, "rise_db": round(rise, 1),
                         "level_db": round(lvl, 1), "nearest_cut_s": round(near, 2) if near is not None else None,
                         "aligned_to_cut": near is not None and near <= 0.15})
    hits.sort(key=lambda h: h["t"])

    # A voice-only track drops close to silence between syllables; a music bed keeps the floor up.
    srt = sorted(full)
    floor, med = srt[len(srt) // 10], srt[len(srt) // 2]
    music = {"music_bed_likely": floor > -48 and med - floor < 18, "floor_p10_db": round(floor, 1),
             "median_db": round(med, 1), "method": "noise floor (p10) vs median level; bed if floor > -48 dB and within 18 dB"}
    return {"status": "ok", "loudness": loudness(video), "music": music, "hits": hits,
            "sfx_candidate_count": len(hits),
            "sfx_aligned_to_cuts": sum(1 for h in hits if h["aligned_to_cut"]),
            "cuts_with_sfx_ratio": round(sum(1 for b in bounds if any(abs(b["t"] - h["t"]) <= 0.15 for h in hits))
                                         / len(bounds), 2) if bounds else 0.0}


# ---------------------------------------------------------------- vision

VISION_PROMPT = """You are analysing short-form video shots so an editor can replicate the edit.
For each shot you get one strip image: [previous shot's last frame | first | mid | last frame of this shot].
Return ONLY a JSON array, one object per shot, with exactly these keys:
{"shot": <int>,
 "framing": "talking_head|b_roll|screen_recording|graphic|text_card|meme_or_stock|product|mixed",
 "shot_size": "extreme_close|close|medium|wide|n/a",
 "composition": "<subject placement, background, aspect use>",
 "camera": "static|pan|tilt|zoom_in|zoom_out|punch_in|handheld|tracking|n/a",
 "on_screen_text": ["<verbatim text other than captions>"],
 "captions": {"present": <bool>, "position": "top|upper_third|center|lower_third|bottom|none",
              "words_per_line": <int or null>, "lines": <int or null>, "font_style": "<e.g. bold sans, all caps, outline>",
              "text_color": "<color>", "highlight_color": "<color of the emphasised/active word or null>",
              "animation_guess": "<word-by-word pop, static, karaoke, none>"},
 "effects": ["<zoom punch, shake, blur, overlay emoji/icon, green screen, split screen, color grade, etc>"],
 "transition_in": {"type": "cut|dissolve|fade|whip_pan|zoom|flash|glitch|slide|match_cut|start",
                   "evidence": "<compare previous last frame with this first frame>"},
 "description": "<one sentence of what is on screen>"}
Compare first vs last frame for camera moves and punch-ins. Be literal; do not invent text you cannot read."""

SUMMARY_PROMPT = """You get a timed transcript, a shot list with vision notes, pacing and audio stats for one short video.
Return ONLY a JSON object:
{"hook": {"first_3s_summary": "<what happens visually and verbally>", "hook_type": "<question|bold claim|visual gag|pattern interrupt|...>"},
 "beats": [{"start": <s>, "end": <s>, "label": "<hook|setup|point 1|example|payoff|cta|...>", "summary": "<one line>"}],
 "format_formula": "<one paragraph recipe for replicating this format>",
 "what_to_steal": ["<concrete, replicable technique with numbers where possible>"]}
Base everything on the data. Do not invent facts."""


def parse_json(text):
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        raise ValueError("no JSON in model reply")
    return json.loads(m.group(1))


def b64(path):
    return base64.standard_b64encode(open(path, "rb").read()).decode()


def ollama_vision_model():
    try:
        tags = json.load(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2))
    except Exception:
        return None
    want = os.environ.get("RAIJINCUT_OLLAMA_MODEL")
    names = [m["name"] for m in tags.get("models", [])]
    if want:
        return want if want in names else None
    return next((n for n in names if re.search(r"llava|vision|-vl|vl:|gemma3|moondream|minicpm-v|qwen2\.5vl", n)), None)


def pick_backend(choice):
    if choice != "auto":
        return choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return "api"
        except ImportError:
            log("  ANTHROPIC_API_KEY is set but the anthropic package is missing (pip install anthropic).")
    if ollama_vision_model():
        return "ollama"
    return "none"


def ask(backend, prompt, images, out):
    """images: list of (label, relpath). Returns reply text."""
    if backend == "api":
        import anthropic
        content = [{"type": "text", "text": prompt}]
        for label, p in images:
            content += [{"type": "text", "text": label},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(os.path.join(out, p))}}]
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("RAIJINCUT_VISION_MODEL", "claude-opus-5"), max_tokens=16000,
            messages=[{"role": "user", "content": content}])
        if msg.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return "".join(b.text for b in msg.content if b.type == "text")
    if backend == "claude-cli":
        listing = "\n".join(f"{label}: {os.path.abspath(os.path.join(out, p))}" for label, p in images)
        full = (prompt + ("\n\nUse the Read tool to view each image file below, then answer. "
                          "Do not use any other tool.\n" + listing if images else ""))
        cmd = ["claude", "-p", full, "--allowedTools", "Read", "--output-format", "text"]
        if os.environ.get("RAIJINCUT_VISION_MODEL"):
            cmd += ["--model", os.environ["RAIJINCUT_VISION_MODEL"]]
        r = run(cmd, cwd=out, timeout=900)
        if r.returncode:
            raise RuntimeError(r.stderr[-500:] or r.stdout[-500:])
        return r.stdout
    if backend == "ollama":
        body = {"model": ollama_vision_model(), "prompt": prompt, "stream": False, "format": "json",
                "images": [b64(os.path.join(out, p)) for _, p in images]}
        req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["response"]
    raise ValueError(backend)


def vision_pass(backend, shots, out, batch=8):
    for i in range(0, len(shots), 1 if backend == "ollama" else batch):
        group = shots[i:i + (1 if backend == "ollama" else batch)]
        imgs = []
        for s in group:
            if s.get("strip"):
                imgs.append((f"Shot {s['index']} ({ts(s['start'])}-{ts(s['end'])}, detector: {s['transition_in']['detected']})", s["strip"]))
            else:
                imgs += [(f"Shot {s['index']} {k} frame", p) for k, p in s["keyframes"].items()]
        log(f"  vision: shots {group[0]['index']}-{group[-1]['index']} via {backend}")
        try:
            res = parse_json(ask(backend, VISION_PROMPT, imgs, out))
            res = res if isinstance(res, list) else [res]
            by = {r.get("shot"): r for r in res if isinstance(r, dict)}
            for k, s in enumerate(group):
                s["vision"] = by.get(s["index"]) or (res[k] if k < len(res) and isinstance(res[k], dict) else None)
        except Exception as e:
            log(f"  vision batch failed: {e}")


def caption_style(shots):
    caps = [s["vision"]["captions"] for s in shots if s.get("vision") and isinstance(s["vision"].get("captions"), dict)]
    if not caps:
        return {"status": "skipped", "todo": "needs a vision backend (see vision.status)"}
    on = [c for c in caps if c.get("present")]
    mode = lambda xs: max(set(xs), key=xs.count) if xs else None
    wpl = [c["words_per_line"] for c in on if isinstance(c.get("words_per_line"), (int, float))]
    return {"status": "ok", "source": "vision", "present_ratio": round(len(on) / len(caps), 2),
            "position": mode([c.get("position") for c in on if c.get("position")]),
            "words_per_line_median": statistics.median(wpl) if wpl else None,
            "lines": mode([c.get("lines") for c in on if c.get("lines")]),
            "font_style": mode([c.get("font_style") for c in on if c.get("font_style")]),
            "text_color": mode([c.get("text_color") for c in on if c.get("text_color")]),
            "highlight_colors": sorted({str(c["highlight_color"]) for c in on if c.get("highlight_color")}),
            "animation": mode([c.get("animation_guess") for c in on if c.get("animation_guess")])}


# ---------------------------------------------------------------- summary

def sentences(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w["text"].endswith((".", "?", "!")):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"start": s[0]["start"], "end": s[-1]["end"], "text": " ".join(w["text"] for w in s)} for s in out]


def heuristic_summary(b):
    words, shots, p, audio = b["transcript"]["words"], b["shots"]["list"], b["pacing"], b["audio"]
    first = [w["text"] for w in words if w["start"] < 3.0]
    sents = sentences(words)
    beats = []
    for i, s in enumerate(sents):
        label = "hook" if i == 0 else "payoff/cta" if i == len(sents) - 1 and len(sents) > 2 else f"beat {i}"
        beats.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "label": label, "summary": s["text"]})
    mix = {}
    for s in shots[1:]:
        t = (s.get("vision") or {}).get("transition_in", {}).get("type") or s["transition_in"]["detected"]
        mix[t] = mix.get(t, 0) + 1
    steal = [f"Cut every {p['avg_shot_s']}s on average ({p['cuts_per_10s']} cuts per 10 s); "
             f"{p['cuts_in_first_3s']} cuts inside the first 3 s."]
    if p["longest_shot"]["duration"] > 2 * p["avg_shot_s"]:
        steal.append(f"Only let one shot breathe: longest is #{p['longest_shot']['index']} at {p['longest_shot']['duration']}s.")
    if audio.get("status") == "ok":
        steal.append(f"{int(audio['cuts_with_sfx_ratio'] * 100)}% of cuts land on an SFX hit "
                     f"({audio['sfx_aligned_to_cuts']} hits on cuts; {audio['sfx_candidate_count']} level jumps overall, most are speech).")
        if audio["music"].get("music_bed_likely"):
            steal.append(f"Keep a music bed under the VO (floor about {audio['music']['floor_p10_db']} dB, "
                         f"{round(audio['music']['median_db'] - audio['music']['floor_p10_db'])} dB under the voice).")
        lufs = audio["loudness"].get("integrated_lufs")
        if lufs is not None:
            steal.append(f"Master to about {lufs} LUFS integrated.")
    cs = b["captions"]
    if cs.get("status") == "ok" and cs["present_ratio"] > 0.3:
        steal.append(f"Burned-in captions at {cs['position']}, about {cs['words_per_line_median']} words per line, "
                     f"{cs['font_style']}; highlight {', '.join(cs['highlight_colors']) or 'none'}.")
    return {"hook": {"first_3s_words": " ".join(first), "first_3s_shots": 1 + p["cuts_in_first_3s"],
                     "first_3s_on_screen_text": sum(((s.get("vision") or {}).get("on_screen_text") or []
                                                     for s in shots if s["start"] < 3.0), [])},
            "beats": beats, "transition_mix": mix, "what_to_steal": steal}


def llm_summary(backend, b, out):
    compact = {
        "transcript": [[round(w["start"], 2), w["text"]] for w in b["transcript"]["words"]],
        "shots": [{k: s.get(k) for k in ("index", "start", "end", "transition_in", "vision")} for s in b["shots"]["list"]],
        "pacing": b["pacing"], "audio": {k: v for k, v in b["audio"].items() if k != "hits"},
        "captions": b["captions"]}
    return parse_json(ask(backend, SUMMARY_PROMPT + "\n\nDATA:\n" + json.dumps(compact), [], out))


# ---------------------------------------------------------------- markdown

def markdown(b):
    src, p, a, s, cs = b["source"], b["pacing"], b["audio"], b["summary"], b["captions"]
    meta = src.get("metadata") or {}
    L = [f"# Breakdown: {meta.get('title') or src['slug']}", ""]
    if meta:
        L += [f"- Source: {meta.get('webpage_url', '')}", f"- Creator: {meta.get('uploader') or meta.get('channel') or 'n/a'}",
              f"- Views / likes: {meta.get('view_count', 'n/a')} / {meta.get('like_count', 'n/a')}"]
    L += [f"- Duration: {src['duration_s']}s, {src['width']}x{src['height']} @ {src['fps']} fps",
          f"- Transcript: {b['transcript']['status']} ({b['transcript'].get('backend') or 'none'}), "
          f"vision: {b['vision']['status']} ({b['vision']['backend']})", ""]
    llm = s.get("llm") or {}
    L += ["## Hook (first 3 s)", ""]
    if llm.get("hook"):
        L += [f"- {llm['hook'].get('first_3s_summary', '')}", f"- Hook type: {llm['hook'].get('hook_type', 'n/a')}"]
    L += [f"- Spoken: \"{s['hook']['first_3s_words']}\"", f"- Shots in first 3 s: {s['hook']['first_3s_shots']}"]
    if s["hook"]["first_3s_on_screen_text"]:
        L.append(f"- On-screen text: {' / '.join(s['hook']['first_3s_on_screen_text'])}")
    L += ["", "## Structure beats", "", "| Time | Beat | What happens |", "|---|---|---|"]
    for bt in llm.get("beats") or s["beats"]:
        L.append(f"| {ts(float(bt['start']))}-{ts(float(bt['end']))} | {bt['label']} | {bt['summary']} |")
    L += ["", "## Cut cadence", "",
          f"- {p['shot_count']} shots, {p['cut_count']} cuts, {p['cuts_per_10s']} cuts per 10 s",
          f"- Average shot {p['avg_shot_s']}s (median {p['median_shot_s']}s, range {p['min_shot_s']}-{p['max_shot_s']}s)",
          f"- Longest shot: #{p['longest_shot']['index']} at {ts(p['longest_shot']['start'])}, {p['longest_shot']['duration']}s",
          "", "## Transition mix", ""]
    L += [f"- {k}: {v}" for k, v in sorted(s["transition_mix"].items(), key=lambda kv: -kv[1])] or ["- none"]
    L += ["", "## SFX and audio", ""]
    if a.get("status") == "ok":
        lo = a["loudness"]
        L += [f"- Loudness: {lo['integrated_lufs']} LUFS integrated, LRA {lo['lra_lu']} LU, peak {lo['true_peak_dbfs']} dBFS",
              f"- Music bed likely: {a['music']['music_bed_likely']} (floor {a['music']['floor_p10_db']} dB, median {a['music']['median_db']} dB)",
              f"- SFX candidates: {a['sfx_candidate_count']}, on cuts: {a['sfx_aligned_to_cuts']} "
              f"({int(a['cuts_with_sfx_ratio'] * 100)}% of cuts have a hit within 150 ms)"]
        for h in [h for h in a["hits"] if h["aligned_to_cut"]][:15]:
            L.append(f"  - {ts(h['t'])} {h['likely']} (+{h['rise_db']} dB{', on cut' if h['aligned_to_cut'] else ''})")
    else:
        L.append(f"- {a.get('status')}")
    L += ["", "## Caption style", ""]
    if cs.get("status") == "ok":
        L += [f"- Present in {int(cs['present_ratio'] * 100)}% of shots, position {cs['position']}, "
              f"{cs['words_per_line_median']} words per line, {cs['lines']} line(s)",
              f"- Font: {cs['font_style']}, color {cs['text_color']}, highlight {', '.join(cs['highlight_colors']) or 'none'}",
              f"- Animation: {cs['animation']}"]
    else:
        L.append(f"- TODO: {cs.get('todo')}")
    L += ["", "## Shot list", "", "| # | Time | Len | In | Framing | Camera | Notes |", "|---|---|---|---|---|---|---|"]
    for sh in b["shots"]["list"]:
        v = sh.get("vision") or {}
        tin = (v.get("transition_in") or {}).get("type") or sh["transition_in"]["detected"]
        L.append(f"| {sh['index']} | {ts(sh['start'])} | {sh['duration']:.2f}s | {tin} | {v.get('framing', '')} | "
                 f"{v.get('camera', '')} | {(v.get('description') or '').replace('|', '/')} |")
    if llm.get("format_formula"):
        L += ["", "## Format formula", "", llm["format_formula"]]
    L += ["", "## What to steal", ""] + [f"- {x}" for x in (llm.get("what_to_steal") or []) + s["what_to_steal"]]
    L += ["", "Contact sheet: `contact_sheet.jpg`. Full data: `breakdown.json`. Subtitles: `transcript.srt`.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def analyze(video, out_dir=".", whisper_model=None, vision="auto", scene_threshold=0.3, no_summary=False, meta_path=None):
    """Write <out_dir>/<slug>/breakdown.json + .md and friends. Returns that folder."""
    video = os.path.abspath(video)
    if not os.path.exists(video):
        sys.exit(f"Video not found: {video}")
    meta_path = meta_path or os.path.splitext(video)[0] + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    slug = slugify(os.path.splitext(os.path.basename(video))[0])
    out = os.path.join(os.path.abspath(out_dir), slug)
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="raijincut-")
    log(f"Analyzing {video}\n  -> {out}")

    info = probe(video)
    b = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "source": {"video_path": video, "slug": slug, **info, "metadata": meta}}

    log("[1/6] transcript")
    words, backend, lang, status = [], None, None, "skipped"
    if info["has_audio"]:
        try:
            backend, lang, words = transcribe_video(video, whisper_model, tmp)
            status = "ok" if backend else "skipped"
        except Exception as e:
            log(f"  transcription failed: {e}")
            status = "failed"
    cues = group_words(words)
    if cues:
        write_srt(cues, os.path.join(out, "transcript.srt"))
    transcript = {"status": status, "backend": backend, "language": lang,
                  "text": " ".join(w["text"] for w in words), "words": words, "cues": cues,
                  "srt": "transcript.srt" if cues else None}
    if status == "skipped":
        transcript["todo"] = "install whisper-cli + pass --whisper-model, or install mlx_whisper"
    json.dump(transcript, open(os.path.join(out, "transcript.json"), "w"), indent=1)
    b["transcript"] = transcript

    log("[2/6] shots")
    frames = frame_stats(video, tmp)
    bounds = detect_boundaries(frames, info["fps"] or 30, scene_threshold,
                               g=thumbs(video))
    shots = build_shots(bounds, info["duration_s"])
    for s in shots:
        s["spoken"] = " ".join(w["text"] for w in words if s["start"] <= w["start"] < s["end"])
    b["shots"] = {"scene_threshold": scene_threshold, "boundaries": bounds, "list": shots}
    b["pacing"] = pacing(shots, info["duration_s"])
    log(f"  {len(shots)} shots, {b['pacing']['cuts_per_10s']} cuts per 10 s")

    log("[3/6] keyframes")
    keyframes(video, shots, out)
    strips(shots, out)
    b["contact_sheet"] = contact_sheet(shots, out)

    log("[4/6] audio")
    b["audio"] = audio_analysis(video, bounds) if info["has_audio"] else {"status": "no_audio"}

    log("[5/6] vision")
    vb = pick_backend(vision)
    b["vision"] = {"backend": vb, "status": "skipped" if vb == "none" else "ok"}
    if vb == "none":
        b["vision"]["todo"] = ("no vision backend: set ANTHROPIC_API_KEY (+ pip install anthropic), run an "
                               "ollama vision model, or pass --vision claude-cli")
    else:
        vision_pass(vb, shots, out)
        if not any(s.get("vision") for s in shots):
            b["vision"]["status"] = "failed"
    b["captions"] = caption_style(shots)

    log("[6/6] summary")
    b["summary"] = heuristic_summary(b)
    if vb in ("api", "claude-cli") and not no_summary:
        try:
            b["summary"]["llm"] = llm_summary(vb, b, out)
        except Exception as e:
            log(f"  LLM summary failed: {e}")
    json.dump(b, open(os.path.join(out, "breakdown.json"), "w"), indent=1)
    open(os.path.join(out, "breakdown.md"), "w").write(markdown(b))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"Done: {os.path.join(out, 'breakdown.md')}")
    return out
