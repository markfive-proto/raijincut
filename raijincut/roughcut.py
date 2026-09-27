"""Transcript-driven rough cut: drop filler words, stutter repeats and long pauses, then join the kept
ranges with one ffmpeg trim/concat pass. This is the only rough-cut implementation; video-studio's
`npm run cleanup` calls `raijincut rough-cut` and adds denoise, loudness and music on top.

Outputs next to OUTPUT: the cut video, <stem>.edits.json (keep ranges, edit list, retimed words)
and <stem>.srt (captions on the new timeline).
"""
import json, os, subprocess, tempfile

from .analyze import group_words, parse_srt, transcribe_video, write_srt

# ponytail: fixed lists. Words that are also real words ("like", "macam", "lah", 那个, 就是) are left
# alone on purpose. BM/ZH fillers only match when the transcript contains them.
FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm",  # EN
    "err", "emm", "eh", "aa", "aaa",  # BM / Manglish
    "嗯", "呃", "额",  # ZH
}


def norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


def load_words(path):
    """Words from JSON: a list or {"words": [...]}, each {"w"|"text"|"word", "start", "end"}."""
    j = json.load(open(path, encoding="utf-8"))
    j = j.get("words", []) if isinstance(j, dict) else j
    return [{"w": (x.get("w") or x.get("text") or x.get("word") or "").strip(), "start": float(x["start"]), "end": float(x["end"])}
            for x in j]


def srt_words(path):
    """Words from an SRT: each cue's words spread evenly over the cue (SRT has no word timings)."""
    words = []
    for a, b, text in parse_srt(path):
        toks = text.split()
        step = (b - a) / max(len(toks), 1)
        words += [{"w": t, "start": a + i * step, "end": a + (i + 1) * step} for i, t in enumerate(toks)]
    return words


def plan_edits(words, pause=0.6, total=None):
    """Decide what to cut. Returns (keep=[{start,end}], edits=[{start,end,reason,text}])."""
    removed = {}
    for i, w in enumerate(words):
        n = norm(w["w"])
        if not n:
            continue
        if n in FILLERS:
            removed[i] = "filler"
        elif i + 1 < len(words) and n == norm(words[i + 1]["w"]):
            removed[i] = "repeat"
        elif i + 3 < len(words) and n == norm(words[i + 2]["w"]) and norm(words[i + 1]["w"]) == norm(words[i + 3]["w"]):
            removed[i] = removed[i + 1] = "repeat"
    edits = [{"start": words[i]["start"], "end": words[i]["end"], "reason": r, "text": words[i]["w"]} for i, r in removed.items()]
    keep, seg, prev = [], None, -1
    for i, w in enumerate(words):
        if i in removed:
            continue
        if seg:
            gap = w["start"] - words[prev]["end"]
            # neighbours join unless the pause is too long; across a removed word only if it left no real gap
            if gap <= (pause if i == prev + 1 else 0.25):
                seg["end"], prev = w["end"], i
                continue
            seg["end"] = min(seg["end"] + 0.12, w["start"] - 0.08)
            keep.append(seg)
            if i == prev + 1:
                edits.append({"start": round(words[prev]["end"], 2), "end": round(w["start"], 2), "reason": "pause", "text": f"{gap:.1f} s"})
        seg = {"start": max(seg["end"] if seg else 0, w["start"] - (0.08 if seg else 0.15)), "end": w["end"]}
        prev = i
    if seg:
        seg["end"] = min(seg["end"] + 0.3, total if total is not None else seg["end"] + 0.3)
        keep.append(seg)
    edits.sort(key=lambda e: e["start"])
    return [{"start": round(k["start"], 3), "end": round(k["end"], 3)} for k in keep], edits


def retime(words, keep):
    """Kept words shifted onto the cut timeline."""
    out, acc = [], 0.0
    for k in keep:
        out += [{**w, "start": round(w["start"] - k["start"] + acc, 3), "end": round(w["end"] - k["start"] + acc, 3)}
                for w in words if w["start"] >= k["start"] and w["end"] <= k["end"]]
        acc += k["end"] - k["start"]
    return out


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def render(src, keep, out):
    """One ffmpeg pass: trim each kept range, concat. .mov/.mkv keep PCM audio for a lossless hand-off."""
    fc = "".join(f"[0:v]trim={k['start']}:{k['end']},setpts=PTS-STARTPTS[v{i}];"
                 f"[0:a]atrim={k['start']}:{k['end']},asetpts=PTS-STARTPTS[a{i}];" for i, k in enumerate(keep))
    fc += "".join(f"[v{i}][a{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=1[v][a]"
    acodec = ["-c:a", "pcm_s16le"] if out.lower().endswith((".mov", ".mkv")) else ["-c:a", "aac", "-b:a", "192k"]
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", *acodec, out], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


def rough_cut(src, out, words_path=None, srt=None, pause=0.6, whisper_model=None):
    if words_path:
        words = load_words(words_path)
    elif srt:
        words = srt_words(srt)
    else:
        with tempfile.TemporaryDirectory(prefix="raijincut-") as tmp:
            backend, _, raw = transcribe_video(src, whisper_model, tmp)
        if not backend:
            raise SystemExit("No transcriber: pass --words/--srt, or install whisper-cli + --whisper-model, or mlx_whisper.")
        words = [{"w": w["text"], "start": w["start"], "end": w["end"]} for w in raw]
    words = [w for w in words if w["w"]]
    if not words:
        raise SystemExit("No words in the transcript; nothing to cut.")
    total = duration(src)
    keep, edits = plan_edits(words, pause, total)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    render(src, keep, out)
    stem = os.path.splitext(out)[0]
    new_words = retime(words, keep)
    write_srt(group_words([{"text": w["w"], "start": w["start"], "end": w["end"]} for w in new_words]), stem + ".srt")
    result = {"source": os.path.abspath(src), "out": os.path.abspath(out), "source_seconds": round(total, 2),
              "cut_seconds": round(sum(k["end"] - k["start"] for k in keep), 2), "pause": pause,
              "keep": keep, "edits": edits, "words": new_words}
    json.dump(result, open(stem + ".edits.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return result
