"""raijincut: video CLI. Every edit is a direct ffmpeg call; analysis lives in analyze.py,
the transcript-driven rough cut in roughcut.py."""
import argparse, json, os, subprocess, sys, tempfile, tomllib

from . import analyze as az
from .roughcut import rough_cut

PRESETS = os.path.join(os.path.dirname(__file__), "presets")
WHISPER_HELP = "whisper.cpp ggml model (env RAIJINCUT_WHISPER_MODEL); without it, mlx_whisper is tried"


def die(msg):
    sys.exit(f"Error: {msg}")


def ff(*args):
    """Run ffmpeg quietly; exit with its error on failure."""
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True)
    if r.returncode:
        die(f"ffmpeg failed:\n{r.stderr.strip()[-1500:]}")


def is_url(s):
    return s.startswith(("http://", "https://"))


# ---------------------------------------------------------------- download

META_FIELDS = "id,title,uploader,channel,webpage_url,extractor_key,duration,view_count,like_count,comment_count,upload_date,description,filepath"
SOCIAL = ("linkedin.com", "facebook.com", "fb.watch", "x.com", "twitter.com", "instagram.com")
LOGIN_WORDS = ("login", "log in", "sign in", "cookies", "authentication", "private", "registered users", "not available")


def download_error(url, stderr, cookies):
    """Readable yt-dlp failure. Login walls (and logged-out social pages) get a sign-in hint."""
    if any(k in stderr.lower() for k in LOGIN_WORDS) or any(h in url for h in SOCIAL):
        hint = ("The browser session did not unlock it. Open the post in that browser while signed in, then retry."
                if cookies else "Sign in to the site in Chrome yourself, then rerun with --cookies-from-browser chrome.")
        return f"This post probably needs a logged-in session. {hint}\n(raijincut never asks for or stores passwords.)\n\n{stderr.strip()}"
    return f"yt-dlp failed:\n{stderr.strip()}"


def download(url, out=".", cookies=None):
    """Best mp4 with audio into `out`, plus <video>.json metadata. Returns the video path."""
    os.makedirs(out, exist_ok=True)
    print(f"Downloading: {url}", flush=True)
    cmd = ["yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b", "--merge-output-format", "mp4",
           "--restrict-filenames", "--no-playlist", "-o", f"{out}/%(title).60B-%(id)s.%(ext)s",
           "--no-simulate", "--print", f"after_move:%(.{{{META_FIELDS}}})j"]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    if r.returncode:
        die(download_error(url, r.stderr, cookies))
    line = next((l for l in reversed(r.stdout.splitlines()) if l.lstrip().startswith("{")), None)
    if not line:
        die("yt-dlp printed no metadata")
    meta = json.loads(line)
    video = meta["filepath"]
    meta["source_url"] = url
    meta_path = os.path.splitext(video)[0] + ".json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"Video:    {video}\nMetadata: {meta_path}")
    return video


# ---------------------------------------------------------------- edits

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    if r.returncode:
        die(r.stderr.strip())
    j = json.loads(r.stdout)
    lines = [f"File: {j['format'].get('filename')}", f"Duration: {j['format'].get('duration')}s"]
    for s in j["streams"]:
        if s["codec_type"] == "video":
            lines.append(f"Video: {s.get('width')}x{s.get('height')}")
        elif s["codec_type"] == "audio":
            lines.append(f"Audio: {s.get('codec_name')}")
    return "\n".join(lines)


def aspect_filter(aspect):
    """Centred crop to W:H with even sizes, as an ffmpeg expression (no probe needed)."""
    try:
        tw, th = (int(x) for x in aspect.split(":"))
    except ValueError:
        die(f"bad aspect ratio '{aspect}', expected W:H like 9:16")
    return f"crop=w='trunc(min(iw,ih*{tw}/{th})/2)*2':h='trunc(min(ih,iw*{th}/{tw})/2)*2'"


def cut(src, start, end, out):
    """Frame-accurate cut (re-encodes; input seeking keeps it fast)."""
    ff("-ss", start, "-to", end, "-i", src, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", out)


def crop(src, out, aspect):
    ff("-i", src, "-vf", aspect_filter(aspect), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", out)


def convert(src, out, crf=23):
    ff("-i", src, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", out)


def font(size):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def burn_subs(src, srt, out, style=None):
    """Burn SRT captions. Works without libass: Pillow draws one transparent PNG per cue, the concat
    demuxer turns them into a timed overlay stream (blank PNG in the gaps), and one ffmpeg overlay
    burns it in. Style keys (preset [subtitle]): font_size, font_color, outline_color, outline_width, offset_y."""
    from PIL import Image, ImageDraw
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", src], capture_output=True, text=True).stdout)
    w, h = j["streams"][0]["width"], j["streams"][0]["height"]
    st = style or {}
    size = int(st.get("font_size") or max(18, round(min(w, h) * 0.055)))
    fnt, color = font(size), st.get("font_color", "#FFFFFF")
    stroke, stroke_w = st.get("outline_color", "#000000"), int(st.get("outline_width", max(2, size // 16)))
    bottom = h - int(st.get("offset_y", round(h * 0.08)))
    cues = sorted(az.parse_srt(srt))
    with tempfile.TemporaryDirectory(prefix="raijincut-subs-") as tmp:
        blank = os.path.join(tmp, "blank.png")
        Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(blank)
        entries, t = [], 0.0
        for i, (a, b, text) in enumerate(cues):
            a = max(a, t)
            if b <= a:
                continue
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            lines, cur = [], ""
            for word in text.replace("\n", " ").split():  # greedy wrap to 85% of the width
                test = f"{cur} {word}".strip()
                if cur and d.textlength(test, font=fnt) > w * 0.85:
                    lines.append(cur)
                    cur = word
                else:
                    cur = test
            lines.append(cur)
            lh = int(size * 1.2)
            for k, ln in enumerate(lines):
                y = bottom - lh * (len(lines) - k)
                d.text(((w - d.textlength(ln, font=fnt)) / 2, y), ln, font=fnt, fill=color,
                       stroke_width=stroke_w, stroke_fill=stroke)
            png = os.path.join(tmp, f"cue{i}.png")
            img.save(png)
            if a > t:
                entries.append((blank, a - t))
            entries.append((png, b - a))
            t = b
        entries += [(blank, 1.0), (blank, 1.0)]  # concat demuxer ignores the last duration
        lst = os.path.join(tmp, "cues.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\nduration {d:.3f}\n" for p, d in entries)
        ff("-i", src, "-f", "concat", "-safe", "0", "-i", lst, "-filter_complex",
           "[1:v]format=rgba[s];[0:v][s]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[v]",
           "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", out)


# ---------------------------------------------------------------- presets / repurpose

def load_preset(name):
    path = os.path.join(PRESETS, f"{name}.toml")
    if not os.path.exists(path):
        die(f"Preset '{name}' not found. Run 'raijincut presets' to see available presets.")
    return tomllib.load(open(path, "rb"))


def list_presets():
    rows = [f"{'PRESET':<15} {'RESOLUTION':<12} {'ASPECT':<10} {'FPS':<8} {'MAX(s)':<8} {'CODEC':<10}", "-" * 65]
    for f in sorted(os.listdir(PRESETS)):
        if not f.endswith(".toml") or f == "default.toml":
            continue
        v = tomllib.load(open(os.path.join(PRESETS, f), "rb")).get("video", {})
        res = f"{v['width']}x{v['height']}" if "width" in v and "height" in v else "-"
        rows.append(f"{f[:-5]:<15} {res:<12} {v.get('aspect_ratio', '-'):<10} {v.get('fps', '-')!s:<8} "
                    f"{v.get('max_duration', '-')!s:<8} {v.get('codec', '-'):<10}")
    return "\n".join(rows)


def repurpose(src, preset, clips, subtitles, out, whisper_model=None):
    """URL or file -> optional clip range -> crop + scale to the preset in one encode -> optional captions."""
    p = load_preset(preset)
    v, sub = p.get("video", {}), p.get("subtitle")
    with tempfile.TemporaryDirectory(prefix="raijincut-") as tmp:
        if is_url(src):
            src = download(src, tmp)
        seek = []
        if clips:
            if ".." not in clips:
                die("--clips must be start..end, e.g. 00:00:10..00:00:30")
            a, b = clips.split("..", 1)
            seek = ["-ss", a, "-to", b]
        vf = aspect_filter(v.get("aspect_ratio", "9:16"))
        if v.get("width") and v.get("height"):
            vf += f",scale={v['width']}:{v['height']}"
        if v.get("fps"):
            vf += f",fps={v['fps']}"
        framed = os.path.join(tmp, "framed.mp4") if subtitles else out
        ff(*seek, "-i", src, "-vf", vf, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", framed)
        if subtitles:
            backend, _, words = az.transcribe_video(framed, whisper_model, tmp)
            if not words:
                print("No transcriber or no speech; saving without subtitles.")
                os.replace(framed, out)
            else:
                srt = os.path.join(tmp, "subs.srt")
                az.write_srt(az.group_words(words), srt)
                burn_subs(framed, srt, out, sub)
    print(f"Repurpose complete: {out}")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="raijincut", description="Video CLI: download, analyze and edit with ffmpeg.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("download", help="download a video (YouTube, X, Facebook, LinkedIn, ...) as mp4 + metadata JSON")
    c.add_argument("url")
    c.add_argument("-o", "--output", default=".")
    c.add_argument("--cookies-from-browser", metavar="BROWSER", help="reuse your signed-in browser session, e.g. chrome")
    c = sp.add_parser("probe", help="print duration, resolution and audio codec")
    c.add_argument("input")
    c = sp.add_parser("cut", help="cut a segment (frame accurate)")
    c.add_argument("input")
    c.add_argument("-s", "--start", required=True)
    c.add_argument("-e", "--end", required=True)
    c.add_argument("-o", "--output", required=True)
    c = sp.add_parser("crop", help="centre-crop to an aspect ratio")
    c.add_argument("input")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("-a", "--aspect", required=True, help="W:H, e.g. 9:16, 16:9, 1:1, 4:5")
    c = sp.add_parser("convert", help="re-encode with a CRF quality")
    c.add_argument("input")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("-c", "--crf", type=int, default=23)
    c = sp.add_parser("subtitle", help="burn SRT subtitles into a video")
    c.add_argument("input")
    c.add_argument("--srt", required=True)
    c.add_argument("-o", "--output", required=True)
    c.add_argument("-p", "--preset", help="take caption style from a preset's [subtitle] table")
    c = sp.add_parser("rough-cut", help="cut filler words, stutter repeats and long pauses (transcript driven)")
    c.add_argument("input")
    c.add_argument("-o", "--output", required=True)
    g = c.add_mutually_exclusive_group()
    g.add_argument("--words", help="word-timed JSON ({words:[{w|text,start,end}]}); default: transcribe")
    g.add_argument("--srt", help="SRT transcript (word times spread evenly per cue)")
    c.add_argument("--pause", type=float, default=0.6, help="longest pause kept, seconds (default 0.6)")
    c.add_argument("--whisper-model", default=os.environ.get("RAIJINCUT_WHISPER_MODEL"), help=WHISPER_HELP)
    c = sp.add_parser("repurpose", help="clip, crop and scale to a platform preset, optional captions")
    c.add_argument("input", help="URL or video file")
    c.add_argument("-p", "--preset", required=True)
    c.add_argument("-c", "--clips", help="start..end, e.g. 00:00:10..00:00:30")
    c.add_argument("--subtitles", action="store_true")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--whisper-model", default=os.environ.get("RAIJINCUT_WHISPER_MODEL"), help=WHISPER_HELP)
    c = sp.add_parser("analyze", help="break a video down: transcript, shots, keyframes, vision, audio, breakdown.md/json")
    c.add_argument("input", help="video file, or a URL to download first")
    c.add_argument("-o", "--output", default=".", help="results go to <output>/<slug>/")
    c.add_argument("--whisper-model", default=os.environ.get("RAIJINCUT_WHISPER_MODEL"), help=WHISPER_HELP)
    c.add_argument("--vision", default="auto", choices=["auto", "api", "ollama", "claude-cli", "none"])
    c.add_argument("--scene-threshold", type=float, default=0.3, help="hard-cut scene score threshold (lower = more cuts)")
    c.add_argument("--cookies-from-browser", metavar="BROWSER")
    c.add_argument("--no-summary", action="store_true", help="skip the LLM summary call")
    sp.add_parser("presets", help="list platform presets")
    a = ap.parse_args(argv)

    if a.cmd == "download":
        download(a.url, a.output, a.cookies_from_browser)
    elif a.cmd == "probe":
        print(probe(a.input))
    elif a.cmd == "cut":
        cut(a.input, a.start, a.end, a.output)
        print(f"Saved: {a.output}")
    elif a.cmd == "crop":
        crop(a.input, a.output, a.aspect)
        print(f"Saved: {a.output}")
    elif a.cmd == "convert":
        convert(a.input, a.output, a.crf)
        print(f"Saved: {a.output}")
    elif a.cmd == "subtitle":
        burn_subs(a.input, a.srt, a.output, load_preset(a.preset).get("subtitle") if a.preset else None)
        print(f"Saved: {a.output}")
    elif a.cmd == "rough-cut":
        r = rough_cut(a.input, a.output, a.words, a.srt, a.pause, a.whisper_model)
        print(f"{r['source_seconds']}s -> {r['cut_seconds']}s, {len(r['edits'])} edits\n{r['out']}")
    elif a.cmd == "repurpose":
        repurpose(a.input, a.preset, a.clips, a.subtitles, a.output, a.whisper_model)
    elif a.cmd == "analyze":
        video = download(a.input, a.output, a.cookies_from_browser) if is_url(a.input) else a.input
        az.analyze(video, a.output, a.whisper_model, a.vision, a.scene_threshold, a.no_summary)
    elif a.cmd == "presets":
        print(list_presets())
