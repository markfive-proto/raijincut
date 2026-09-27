# RaijinCut

> Named after Raijin, the Japanese god of thunder.

A small Python CLI for short-form video work: download a reference, break it down (transcript, shots,
transitions, pacing, audio, captions), and edit with direct ffmpeg calls (cut, crop, convert, burn
captions, transcript-driven rough cut, platform repurpose).

Design rules:
1. **No moviepy.** Every edit is one ffmpeg command.
2. **No PyTorch.** Transcripts come from whisper.cpp (`whisper-cli`) or `mlx_whisper`. No API key is needed.
3. **Low RAM.** Frames and audio are streamed from ffmpeg pipes one small chunk at a time (32 px gray
   frames for the dissolve test, 160 px for scene scores, 10 ms audio hops). Full-resolution frames are
   never loaded into Python.

## Commands

| Command | What it does |
|---|---|
| `download` | Download a video (YouTube, X, Facebook, LinkedIn) as mp4 + metadata JSON; optional browser cookies |
| `analyze` | Break a video (file or URL) down: word-level transcript + SRT, shots and transitions, pacing, keyframes, contact sheet, vision notes, audio/SFX hits, caption style, `breakdown.json` + `breakdown.md`. `--mode motion`: dense frames, per-shot easing, camera, holds, transition sheets, animation specs, `motion.json` + `motion.md` |
| `probe` | Duration, resolution, audio codec |
| `cut` | Cut a segment by start/end (frame accurate, re-encodes) |
| `crop` | Centre-crop to any `W:H` aspect ratio |
| `convert` | Re-encode with a CRF quality |
| `subtitle` | Burn SRT captions into a video (works on ffmpeg builds without libass) |
| `rough-cut` | Transcript-driven cleanup: removes filler words, stutter repeats and long pauses |
| `repurpose` | URL or file, optional clip range, crop + scale to a platform preset, optional captions |
| `presets` | List platform presets |

## Install

```bash
uv tool install -e ~/Projects/raijincut      # or: pipx install -e ~/Projects/raijincut
raijincut --help
```

Python 3.11+. The only Python dependency is Pillow (caption and contact-sheet drawing); `anthropic`
is optional (`uv tool install -e '.[api]'`) for `--vision api`.

| Tool | Needed for | Install |
|---|---|---|
| ffmpeg + ffprobe | everything | `brew install ffmpeg` |
| yt-dlp | `download`, URLs in `analyze` / `repurpose` | `brew install yt-dlp` |
| whisper.cpp + a ggml model, or mlx_whisper | transcripts (`analyze`, `rough-cut` without `--words`, `repurpose --subtitles`) | `brew install whisper-cpp` + e.g. `ggml-small.en.bin`; or `pip install mlx-whisper` |

Set `RAIJINCUT_WHISPER_MODEL=/path/to/ggml-small.en.bin` once instead of passing `--whisper-model`.
`RAIJINCUT_MLX_MODEL` picks the mlx model when whisper.cpp is not used.

## Usage

### Download

```bash
raijincut download "https://youtube.com/shorts/..." -o ./downloads

# LinkedIn / X / Facebook posts usually need a signed-in session.
# Sign in to the site in Chrome yourself, then reuse that session:
raijincut download "https://www.linkedin.com/posts/..." -o ./downloads --cookies-from-browser chrome
```

Writes `<title>-<id>.mp4` (best mp4 with audio) and `<title>-<id>.json` next to it with `id`, `title`,
`uploader`, `channel`, `webpage_url`, `source_url`, `extractor_key`, `duration`, `view_count`, `like_count`,
`comment_count`, `upload_date`, `description` and `filepath` (counts are `null` when the site hides them).
raijincut never asks for, types or stores passwords. If a login wall blocks the download it prints a
message telling you to sign in in Chrome and rerun with `--cookies-from-browser chrome`.

### Analyze a reference video

```bash
# a local file, or a URL (downloaded first)
raijincut analyze ./downloads/video.mp4 -o ./breakdowns
raijincut analyze "https://youtube.com/shorts/..." -o ./breakdowns \
  --whisper-model ~/models/ggml-small.en.bin --vision auto
```

Everything lands in `<output>/<slug>/`:

| File | What |
|---|---|
| `breakdown.md` | Human summary: hook in the first 3 s, structure beats, cut cadence, transition mix, SFX usage, caption style, shot list, what to steal |
| `breakdown.json` | All data, stable schema below |
| `transcript.json`, `transcript.srt` | Word-level transcript and caption-sized SRT cues |
| `keyframes/shotNNN_{first,mid,last}.jpg` | Three frames per shot |
| `strips/shotNNN.jpg` | `[previous shot's last frame, first, mid, last]` in one image (what the vision pass sees) |
| `contact_sheet.jpg` | Mid frame of every shot with index, start time and length |

Options and backends:

| Section | How | Needs |
|---|---|---|
| Transcript | whisper.cpp (`whisper-cli -ml 1 -sow`) with `--whisper-model` / `RAIJINCUT_WHISPER_MODEL`; otherwise `mlx_whisper --word-timestamps` (`RAIJINCUT_MLX_MODEL` picks the model); otherwise skipped with a TODO | `whisper-cli` + a ggml model, or `mlx_whisper` |
| Shots | ffmpeg `scene` score spikes = hard cuts (`--scene-threshold`, default 0.3); luma runs = fade through black / white flash; pixel blend test on 32 px frames = dissolves | ffmpeg |
| Keyframes | ffmpeg; strips and contact sheet use Pillow | ffmpeg, Pillow |
| Vision | `--vision auto` uses the Claude API when `ANTHROPIC_API_KEY` is set (`pip install anthropic`, model `RAIJINCUT_VISION_MODEL`, default `claude-opus-5`), else a local ollama vision model (`RAIJINCUT_OLLAMA_MODEL`), else skips with a TODO. `--vision claude-cli` uses the local `claude` CLI (your Claude Code login) instead. Keys come from the environment only | one of the above |
| Audio | ffmpeg `ebur128` loudness; 10 ms envelopes in three bands (full, >3 kHz, <150 Hz) for SFX hits; noise floor vs median for a music bed guess | ffmpeg |
| Captions | Burned-in caption style aggregated from the vision pass | a vision backend |
| Summary | Heuristic hook/beats/what-to-steal always; with `api` or `claude-cli` also an LLM pass for beats, format formula and what to steal (`--no-summary` skips it) | optional |

Memory stays flat with video length: scene scores come from a 160 px ffmpeg pass, the dissolve test keeps only a one-second window of 32 px gray frames, and audio envelopes are read 10 ms at a time. On the 59 s sample, Python peaked at about 55 MB; the biggest child process is whisper-cli with its model.

#### `breakdown.json` schema (version 1)

Times are seconds from the start of the video. Fields a backend could not fill are `null`, and a skipped
section has `"status": "skipped"` plus a `"todo"`.

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-09-27T09:00:00",
  "source": {
    "video_path": "/abs/path.mp4", "slug": "big-o-explained-...",
    "duration_s": 59.0, "width": 608, "height": 1080, "fps": 23.976, "has_audio": true,
    "metadata": { /* the download JSON above, or null for local files */ }
  },
  "transcript": {
    "status": "ok|skipped|failed", "backend": "whisper.cpp:ggml-small.en.bin", "language": "en",
    "text": "...",
    "words": [{"text": "If", "start": 0.03, "end": 0.10}],
    "cues": [{"start": 0.03, "end": 2.88, "text": "If you want to get a job ..."}],
    "srt": "transcript.srt"
  },
  "shots": {
    "scene_threshold": 0.3,
    "boundaries": [{"t": 1.585, "type": "cut|dissolve|fade_black|cut_to_black|flash_white", "confidence": 0.9, "score": 0.75}],
    "list": [{
      "index": 1, "start": 0.0, "end": 1.585, "duration": 1.585,
      "transition_in": {"detected": "start|cut|dissolve|fade_black|cut_to_black|flash_white", "confidence": 1.0},
      "spoken": "words whose start falls in this shot",
      "keyframes": {"first": "keyframes/shot001_first.jpg", "mid": "...", "last": "..."},
      "strip": "strips/shot001.jpg",
      "vision": {                                  // null when vision was skipped
        "shot": 1,
        "framing": "talking_head|b_roll|screen_recording|graphic|text_card|meme_or_stock|product|mixed",
        "shot_size": "extreme_close|close|medium|wide|n/a",
        "composition": "...",
        "camera": "static|pan|tilt|zoom_in|zoom_out|punch_in|handheld|tracking|n/a",
        "on_screen_text": ["..."],
        "captions": {"present": true, "position": "top|upper_third|center|lower_third|bottom|none",
                     "words_per_line": 3, "lines": 1, "font_style": "...", "text_color": "...",
                     "highlight_color": "... or null", "animation_guess": "..."},
        "effects": ["..."],
        "transition_in": {"type": "cut|dissolve|fade|whip_pan|zoom|flash|glitch|slide|match_cut|start", "evidence": "..."},
        "description": "..."
      }
    }]
  },
  "pacing": {
    "shot_count": 27, "cut_count": 26, "avg_shot_s": 2.18, "median_shot_s": 1.8,
    "min_shot_s": 0.25, "max_shot_s": 4.5, "cuts_per_10s": 4.4, "cuts_in_first_3s": 2,
    "longest_shot": {"index": 15, "start": 27.99, "duration": 4.5}
  },
  "contact_sheet": "contact_sheet.jpg",
  "audio": {
    "status": "ok|no_audio",
    "loudness": {"integrated_lufs": -19.8, "lra_lu": 2.2, "true_peak_dbfs": -4.3},
    "music": {"music_bed_likely": true, "floor_p10_db": -37.4, "median_db": -23.9, "method": "..."},
    "hits": [{"t": 2.99, "band": "full|high|low", "likely": "hit|whoosh_or_click|impact",
              "rise_db": 18.5, "level_db": -20.1, "nearest_cut_s": 0.15, "aligned_to_cut": true}],
    "sfx_candidate_count": 40, "sfx_aligned_to_cuts": 12, "cuts_with_sfx_ratio": 0.46
  },
  "vision": {"backend": "api|ollama|claude-cli|none", "status": "ok|skipped|failed", "todo": "only when skipped"},
  "captions": {
    "status": "ok|skipped", "source": "vision", "present_ratio": 0.8, "position": "center",
    "words_per_line_median": 3, "lines": 1, "font_style": "...", "text_color": "...",
    "highlight_colors": ["..."], "animation": "..."
  },
  "summary": {
    "hook": {"first_3s_words": "...", "first_3s_shots": 3, "first_3s_on_screen_text": ["..."]},
    "beats": [{"start": 0.03, "end": 2.88, "label": "hook", "summary": "..."}],
    "transition_mix": {"cut": 24, "dissolve": 1},
    "what_to_steal": ["Cut every 2.2s on average ..."],
    "llm": {                                       // only with --vision api or claude-cli
      "hook": {"first_3s_summary": "...", "hook_type": "..."},
      "beats": [{"start": 0.0, "end": 3.0, "label": "hook", "summary": "..."}],
      "format_formula": "...",
      "what_to_steal": ["..."]
    }
  }
}
```

Heuristics worth knowing: SFX hits are level jumps, so speech plosives and sibilants can show up too
(hits with `aligned_to_cut: true` are the reliable ones). The music-bed flag compares the noise floor to
the median level, so a noisy room can read as music. Detected transitions come from pixels; the vision
pass gives a second opinion in `vision.transition_in`.

### Motion mode: how a video animates

```bash
raijincut analyze ./downloads/launch.mp4 -o ./breakdowns --mode motion --vision claude-cli
raijincut analyze ./downloads/launch.mp4 -o ./breakdowns --mode motion --dense-fps 15 --vision none
```

For motion-design references (SaaS launch films, product reels): what moves, how far, how long, with
which easing, how the camera moves and how each shot hands over to the next. No transcript. Output in
`<output>/<slug>/`:

| File | What |
|---|---|
| `motion.md` | Per video: computed easing, camera, transition and sound-sync stats, then per shot the computed moves and (with vision) an animation spec table |
| `motion.json` | All data, schema below |
| `dense/fNNNNN.jpg` | Every frame at `--dense-fps` (default 10), 480 px wide; frame `fK` is at `(K-1)/fps` s |
| `strips/shotNNN.jpg` | Up to 8 dense frames spread over the shot, labelled with the time from the shot start |
| `transitions/cutNNN.jpg` | 30 frames at 30 fps from 0.5 s before to 0.5 s after cut NNN, labelled in ms, the first frame after the cut outlined red |

How the numbers are made (all computed from pixels, no guessing):

- **Energy**: mean absolute difference between consecutive 64 px gray frames at `--dense-fps`. Holds are
  runs under 0.5 (or 10% of the shot's peak move) of at least 0.3 s; the rest is split into motion events.
- **Global motion**: phase correlation between consecutive 64 px frames (pure Python FFT) gives the shift;
  the divergence of the four quadrant shifts gives the scale change (it reads about a fifth low). An event is a
  `camera` move when the whole frame moves (quadrants agree and at least 20% of pixels change) or scales, a
  `fade` when the luma change explains the difference, else an `element` move.
- **Easing**: the event's speed curve (phase correlation speed when it locks on, else energy) is split into
  pulses. One pulse: flat top = `linear`, mass early = `ease-out`, late = `ease-in`, bell = `ease-in-out`.
  Decaying pulses or a direction flip = `overshoot` (2) or `bounce` (3+); similar pulses = `staggered <ease>`
  with the spacing as `stagger_s`. `confidence` is a heuristic 0-1.
- **Transitions**: the breakdown detector (cut, dissolve, fade through black, white flash), then for hard cuts
  the 30 fps window decides `zoom_through` (scale change next to the cut), `whip` (fast shift), `blur`
  (sharpness dip), `animated` (4+ changing frames that are not a blend: a wipe, push or morph) or `cut`.
- **Sound sync**: audio onsets (10 dB level jumps) within 100 ms of a cut or a move start, with the chance
  rate for that onset density.
- **Vision** (`--vision claude-cli|api|ollama`, `--vision-batch 4` shots per call): the strip, the transition
  sheet and the computed signals go in; per shot it returns elements, animations (element, property,
  direction, from, to, start, duration, stagger, easing), camera, transition out, typography, UI presentation,
  background, effects and sound sync.

Self-check (`tests/test_raijincut.py`): a lavfi clip with a card sliding in on a cubic ease-out, a linear pan, an
eased zoom, a hard cut and a fade through black; all 9 checks (shots, easings, move kind, camera, transitions)
must pass. A 60 s 1080p video takes about a minute before vision; Python stays under 100 MB.

#### `motion.json` schema (version 1)

```jsonc
{
  "schema_version": 1, "mode": "motion", "generated_at": "2026-09-27T09:00:00",
  "source": { /* as in breakdown.json */ },
  "settings": {"dense_fps": 10, "dense_width": 480, "signal_size": 64, "window_s": 0.5, "window_fps": 30, "scene_threshold": 0.3},
  "pacing": { /* as in breakdown.json */ },
  "signals": {"fps": 10, "energy": [0.0, 3.7], "coverage": [], "luma": [], "dluma": [], "dx": [], "dy": [],
              "zoom": [], "pc_peak": [], "agree": []},   // sample k is at k/fps, values describe the change from k-1
  "audio": {"status": "ok|no_audio", "onsets": [1.23], "onsets_per_s": 1.5, "cuts_on_onset": 4,
            "events_on_onset": 9, "event_count": 30, "chance_ratio": 0.3},
  "transitions": [{"index": 1, "t": 2.5, "detected": "cut|dissolve|fade_black|cut_to_black|flash_white",
                   "computed": "cut|zoom_through|whip|blur|animated|dissolve|fade_black|cut_to_black|flash_white",
                   "confidence": 0.9, "on_onset": true, "sheet": "transitions/cut001.jpg",
                   "metrics": {"peak_diff": 113.7, "changing_frames": 1, "max_speed": 0.01, "max_zoom": 0.0, "blur_ratio": 0.9, "luma_min": 46.5}}],
  "shots": [{
    "index": 1, "start": 0.0, "end": 2.5, "duration": 2.5, "transition_in": {"detected": "start", "confidence": 1.0},
    "transition_out": 1,                                   // index into transitions, null for the last shot
    "strip": "strips/shot001.jpg", "dense_frames": ["dense/f00001.jpg", "dense/f00025.jpg"],
    "motion": {
      "camera": ["static|pan_left|pan_right|tilt_up|tilt_down|zoom_in|zoom_out"],
      "holds": [[1.2, 2.4]], "hold_ratio": 0.48, "energy_peak": 13.2, "energy_mean": 3.1, "samples": 24,
      "main_event": 0,                                     // biggest non-fade move
      "events": [{"start": 0.4, "end": 1.2, "duration": 0.8, "peak_t": 0.7,
                  "kind": "element|camera|fade", "direction": "left|right|up|down|in|out|n/a",
                  "travel": [0.55, 0.0],                   // content shift as fractions of frame width, height
                  "scale": 1.0, "agree": 0.25, "coverage": 0.04, "energy": 64.9,
                  "speed_source": "phase_correlation|energy",
                  "easing": "linear|ease-out|ease-in|ease-in-out|overshoot|bounce|staggered <ease>|instant",
                  "confidence": 0.86, "pulses": 1, "stagger_s": null, "centroid": 0.27, "plateau": 0.29,
                  "on_onset": false}]
    },
    "vision": {                                            // null without a vision backend
      "shot": 1, "role": "hook|problem|solution|demo|feature|social_proof|cta|brand|transition|other",
      "elements": ["..."],
      "animations": [{"element": "...", "property": "position|scale|opacity|blur|rotation|mask|clip|colour|3d|path|text|counter",
                      "direction": "...", "from": "...", "to": "...", "start_s": 0.4, "duration_s": 0.8,
                      "stagger_s": null, "easing": "..."}],
      "camera": {"move": "static|push_in|pull_out|pan|tilt|orbit|dolly|parallax|shake|rack_focus", "direction": "...", "amount": "...", "easing": "..."},
      "transition_out": {"type": "cut|match_cut|whip|zoom_through|mask_wipe|morph|push|slide|dissolve|fade|blur|light_leak|flash|glitch|none",
                         "duration_s": 0.3, "evidence": "..."},
      "typography": {"present": true, "text": ["..."], "weight": "...", "size_class": "hero|headline|body|label|none",
                     "kinetic": "none|word_by_word|char_cascade|mask_reveal|scale_pop|typewriter|highlight|counter|other"},
      "ui_presentation": "real_screen_recording|rebuilt_ui|device_mockup|3d_device|illustration|live_action|none",
      "background": {"type": "flat|gradient|mesh_gradient|glow|glass|grid|photo|video|3d", "colours": ["..."], "motion": "..."},
      "effects": ["..."], "sound_sync": "...", "notes": "..."
    }
  }],
  "vision": {"backend": "claude-cli|api|ollama|none", "status": "ok|skipped|failed"},
  "summary": {"easing_main_events": {"ease-out": 12}, "easing_all_events": {}, "event_kinds": {}, "event_duration_median_s": 0.6,
              "stagger_median_s": 0.1, "camera": {}, "transitions_computed": {}, "hold_ratio": 0.3,
              "vision": {"roles": [], "transitions": {}, "camera": {}, "animation_properties": {}, "animation_easing": {},
                         "animation_duration_median_s": 0.5, "animation_stagger_median_s": 0.08, "ui_presentation": {},
                         "background": {}, "kinetic_type": {}, "effects": {}}}
}
```

### Rough cut (filler words, repeats, pauses)

```bash
raijincut rough-cut talk.mp4 -o clean.mp4                      # transcribes first
raijincut rough-cut talk.mp4 -o clean.mp4 --words words.json   # reuse a word-timed transcript
raijincut rough-cut talk.mp4 -o clean.mp4 --srt talk.srt       # or an SRT (word times spread per cue)
raijincut rough-cut talk.mp4 -o clean.mov --pause 0.8          # .mov/.mkv keep PCM audio for further processing
```

Removes filler words (`um`, `uh`, `erm`, `ah`, `hmm`, BM `err`/`emm`/`eh`, ZH `嗯`/`呃`/`额`), stutter repeats
(`we we`, `I think I think`) and pauses longer than `--pause` seconds, then joins the kept ranges in one
ffmpeg trim/concat pass. Words that are also real words (`like`, `actually`, `lah`) are never cut.
`--words` takes `{"words": [{"w" or "text", "start", "end"}]}` (the `analyze` transcript.json works).

Next to the output: `<name>.edits.json` (keep ranges, every edit with its reason, words retimed to the
new timeline) and `<name>.srt` (captions on the new timeline). This is the single rough-cut
implementation: `npm run cleanup` in solopreneur's video-studio calls it, then adds denoise, loudness and
a ducked music bed.

### Cut, crop, convert, captions

```bash
raijincut probe video.mp4
raijincut cut video.mp4 -s 00:00:30 -e 00:02:00 -o clip.mp4
raijincut crop video.mp4 -a 9:16 -o vertical.mp4
raijincut convert input.mp4 -o output.mp4 -c 18
raijincut subtitle video.mp4 --srt subs.srt -o final.mp4 [-p shorts]
```

`subtitle` draws each cue with Pillow (white text, black outline, bottom of the frame; `-p <preset>`
uses that preset's `[subtitle]` size, colours and offset) and burns them with a single ffmpeg overlay,
so it works on ffmpeg builds without libass or drawtext.

### Repurpose for a platform

```bash
raijincut repurpose "https://youtube.com/watch?v=..." -p tiktok -c 00:00:30..00:01:00 --subtitles -o tiktok.mp4
raijincut repurpose talk.mp4 -p reels -o reel.mp4
raijincut presets
```

One encode for clip + crop + scale + fps; `--subtitles` transcribes the result and burns captions in the
preset's style.

## Platform presets

TOML files in `raijincut/presets/` define resolution, aspect ratio, fps, duration limit and caption style:

| Preset | Resolution | Aspect | FPS | Max duration |
|--------|-----------|--------|-----|-------------|
| tiktok | 1080x1920 | 9:16 | 30 | 180s |
| reels | 1080x1920 | 9:16 | 30 | 90s |
| shorts | 1080x1920 | 9:16 | 30 | 60s |
| linkedin | 1080x1350 | 4:5 | 30 | 300s |
| twitter | 1280x720 | 16:9 | 30 | 140s |
| square | 1080x1080 | 1:1 | 30 | 60s |

## Layout

```
raijincut/
├── cli.py        # argparse entry point; download, probe, cut, crop, convert, subtitle, repurpose, presets
├── analyze.py    # analyze: transcript, shots, keyframes, vision, audio, breakdown (+ shared SRT/whisper helpers)
├── motion.py     # analyze --mode motion: dense frames, energy, phase correlation, easing, transitions, motion.json/md
├── roughcut.py   # rough-cut: plan edits from a transcript, one ffmpeg trim/concat pass
└── presets/      # platform presets (TOML)
tests/test_raijincut.py   # one self-check per command on ffmpeg lavfi samples
```

Tests: `python3 tests/test_raijincut.py` (or `python -m pytest tests`). No network, whisper or vision needed.

## License

MIT
