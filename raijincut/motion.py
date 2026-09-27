"""`raijincut analyze --mode motion`: how a video animates, shot by shot. Writes motion.json + motion.md.

One ffmpeg pass samples the video at 10 fps: 480 px JPGs for the eye (dense/) and 64 px gray frames for
the maths, streamed one frame at a time. Per shot: a frame-difference energy curve, global motion from
phase correlation (pan from the whole frame, zoom from the four quadrants), holds, and motion events with
an easing guess. Per cut: a 30 fps window of +-0.5 s as a transition sheet. An optional vision pass sends
frame strips plus these numbers and asks for per-shot animation specs as JSON.
Pure Python (no numpy): the FFT below is small enough, and 64 px frames keep it fast.
"""
import cmath, datetime, json, math, os, shutil, statistics, sys
from collections import Counter
from . import analyze as az

SCHEMA_VERSION = 1
N = 64          # signal frames are N x N gray; aspect is squashed, so shifts are reported as fractions of the frame
Q = N // 2      # quadrant size for the zoom estimate
QUADS = ((0, 0, -Q / 2, -Q / 2), (Q, 0, Q / 2, -Q / 2), (0, Q, -Q / 2, Q / 2), (Q, Q, Q / 2, Q / 2))  # x0, y0, centre dx, dy

# ponytail: fixed thresholds tuned on the lavfi self-checks and a few real videos; the vision pass is the second opinion.
ACTIVE_MIN = 0.5        # energy (mean abs diff, 0-255) below this is a hold, whatever the shot's peak
ACTIVE_REL = 0.1        # ... and below 10% of the shot's peak energy
HOLD_MIN_S = 0.3
SYNC_S = 0.1            # an audio onset this close to a cut or motion start counts as in sync


# ---------------------------------------------------------------- phase correlation

_TW, _HANN = {}, {}


def fft(a, inv=False):
    """Iterative radix-2 FFT of a list of complex numbers (length a power of two). The inverse is unscaled."""
    n = len(a)
    a = list(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        half = size // 2
        tw = _TW.get((size, inv))
        if tw is None:
            tw = _TW[(size, inv)] = [cmath.exp((2j if inv else -2j) * math.pi * k / size) for k in range(half)]
        for s in range(0, n, size):
            for k in range(half):
                u, v = a[s + k], a[s + k + half] * tw[k]
                a[s + k], a[s + k + half] = u + v, u - v
        size *= 2
    return a


def fft2(rows, inv=False):
    """2-D FFT. Forward takes [y][x] and returns [kx][ky]; the inverse of that layout returns [y][x]."""
    return [fft(c, inv) for c in zip(*[fft(r, inv) for r in rows])]


def hann(n):
    if n not in _HANN:
        _HANN[n] = [0.5 - 0.5 * math.cos(2 * math.pi * (i + 0.5) / n) for i in range(n)]
    return _HANN[n]


def spectrum(g, x0, y0, n):
    """FFT of the n x n patch at (x0, y0) of an N x N gray frame, mean removed and Hann windowed."""
    w = hann(n)
    rows = [g[(y0 + y) * N + x0:(y0 + y) * N + x0 + n] for y in range(n)]
    m = sum(map(sum, rows)) / (n * n)
    return fft2([[(v - m) * w[x] * w[y] for x, v in enumerate(r)] for y, r in enumerate(rows)])


def phase_corr(A, B, n):
    """(dx, dy, peak): the shift in pixels that moves patch A onto patch B, and the correlation peak
    (0-1; near 1 for a pure shift, low for flat frames, blur or unrelated content)."""
    R = []
    for ra, rb in zip(A, B):
        row = []
        for a, b in zip(ra, rb):
            c = b * a.conjugate()
            m = abs(c)
            row.append(c / m if m > 1e-9 else 0j)
        R.append(row)
    r = fft2(R, inv=True)
    best, py, px = -1e9, 0, 0
    for y, row in enumerate(r):
        for x, v in enumerate(row):
            if v.real > best:
                best, py, px = v.real, y, x

    def sub(l, c, rr):  # parabolic sub-pixel offset of a peak
        d = l - 2 * c + rr
        return 0.5 * (l - rr) / d if d < 0 else 0.0
    dx = px + sub(r[py][px - 1].real, best, r[py][(px + 1) % n].real)
    dy = py + sub(r[py - 1][px].real, best, r[(py + 1) % n][px].real)
    dx = dx - n if dx > n / 2 else dx
    dy = dy - n if dy > n / 2 else dy
    return dx, dy, max(0.0, best / (n * n))


def spectra(g):
    return [spectrum(g, 0, 0, N)] + [spectrum(g, x0, y0, Q) for x0, y0, _, _ in QUADS]


def global_motion(sa, sb):
    """Pan from the whole frame; zoom as the divergence of the four quadrant shifts.
    Returns (dx, dy, zoom, peak, agree): dx, dy in pixels of the N px frame, zoom = scale change - 1
    (approximate; it reads low by about a fifth), agree = share of quadrants that moved with the whole frame
    (near 1 for a camera move over a textured frame, low for one element moving over a flat background)."""
    dx, dy, peak = phase_corr(sa[0], sb[0], N)
    qs = [phase_corr(a, b, Q) for a, b in zip(sa[1:], sb[1:])]
    zoom = 0.0
    if min(q[2] for q in qs) >= 0.08:
        mx, my = sum(q[0] for q in qs) / 4, sum(q[1] for q in qs) / 4
        zoom = sum((q[0] - mx) * cx + (q[1] - my) * cy for q, (_, _, cx, cy) in zip(qs, QUADS)) / (4 * 2 * (Q / 2) ** 2)
    agree = sum(1 for q in qs if q[2] >= 0.1 and math.hypot(q[0] - dx, q[1] - dy) <= 1.0) / 4
    return dx, dy, zoom, peak, agree


# ---------------------------------------------------------------- easing

def _shape(p):
    """Easing of one speed pulse (normalised to peak 1): where its mass sits and how flat its top is."""
    m = max(p)
    if m <= 0:
        return "instant", 0.5, None, None
    p = [x / m for x in p]
    tot = sum(p)
    c = sum(i * x for i, x in enumerate(p)) / tot / max(1, len(p) - 1)
    core = [i for i, x in enumerate(p) if x >= 0.1]
    core = p[core[0]:core[-1] + 1]                  # the quiet samples either side do not count against a flat top
    plateau = sum(1 for x in core if x >= 0.75) / len(core)
    asym = 0.5 - c
    if plateau >= 0.7:
        return "linear", min(0.95, 0.55 + (plateau - 0.7) * 1.3), round(c, 3), round(plateau, 3)
    if asym >= 0.07:
        return "ease-out", min(0.95, 0.55 + 2 * (asym - 0.07)), round(c, 3), round(plateau, 3)
    if asym <= -0.07:
        return "ease-in", min(0.95, 0.55 + 2 * (-asym - 0.07)), round(c, 3), round(plateau, 3)
    return "ease-in-out", min(0.9, 0.5 + 3 * (0.07 - abs(asym)) + 0.5 * (0.7 - plateau)), round(c, 3), round(plateau, 3)


def pulses(p, dip=0.5, rise=0.08):
    """Split a speed curve into pulses at valleys that fall below `dip` x the pulse's peak and rise again."""
    out, start, peak, lo = [], 0, p[0], None    # lo: index of the valley we are in, None while not in one
    for i in range(1, len(p)):
        x = p[i]
        if lo is None:
            if x > peak:
                peak = x
            elif x < dip * peak:
                lo = i
        elif x < p[lo]:
            lo = i
        elif x >= p[lo] + rise:
            out.append((start, lo))
            start, peak, lo = lo, x, None
    out.append((start, len(p) - 1))
    return out


def reversals(vel, rel=0.05):
    """Sign flips of a signed velocity, ignoring samples under `rel` x its peak magnitude and single-sample
    flips (phase-correlation noise); a real overshoot holds the reversed direction for 2+ samples."""
    m = max((abs(x) for x in vel), default=0)
    s = [x > 0 for x in vel if m and abs(x) >= rel * m]
    runs_ = []
    for x in s:
        if runs_ and runs_[-1][0] == x:
            runs_[-1][1] += 1
        else:
            runs_.append([x, 1])
    kept = [r[0] for i, r in enumerate(runs_) if r[1] >= 2 or i == 0]
    return sum(a != b for a, b in zip(kept, kept[1:]))


def classify_easing(v, fps=10, vel=None):
    """Easing feel of one motion event from its speed (energy) curve sampled at `fps`.
    One pulse: linear (flat top), ease-out (mass early), ease-in (mass late), ease-in-out (bell).
    Several pulses: decaying ones are a settle (2 = overshoot, 3+ = bounce); similar ones are a stagger.
    `vel` (signed velocity along the move, optional) catches an overshoot too small to show in the energy.
    Returns {easing, confidence, pulses, stagger_s, centroid, plateau}."""
    if len(v) < 3 or max(v) <= 0:
        return {"easing": "instant", "confidence": 0.5, "pulses": 1, "stagger_s": None, "centroid": None, "plateau": None}
    m = max(v)
    p = [x / m for x in v]
    ps = pulses(p)
    peaks = [max(p[a:b + 1]) for a, b in ps]
    first = p[ps[0][0]:ps[0][1] + 1]
    ease, conf, c, pl = _shape(first if len(ps) > 1 else p)
    res = {"pulses": len(ps), "stagger_s": None, "centroid": c, "plateau": pl}
    flips = reversals(vel) if vel else 0
    if len(ps) == 1 and flips and ease == "ease-out":
        ease, conf = ("overshoot" if flips == 1 else "bounce"), 0.6
    elif len(ps) > 1:
        decaying = all(b < 0.9 * a for a, b in zip(peaks, peaks[1:])) and peaks[-1] < 0.6 * peaks[0]
        if decaying or flips >= len(ps) - 1:   # a settle reverses direction between pulses; a stagger does not
            ease = "overshoot" if len(ps) == 2 else "bounce"
            conf = min(0.9, 0.55 + 0.1 * len(ps) + 0.3 * (1 - peaks[1] / peaks[0]))
        else:
            tops = [a + max(range(b - a + 1), key=lambda k: p[a + k]) for a, b in ps]
            res["stagger_s"] = round(statistics.mean(y - x for x, y in zip(tops, tops[1:])) / fps, 3)
            ease = f"staggered {ease}"
            conf *= 0.8
    return {"easing": ease, "confidence": round(conf, 2), **res}


# ---------------------------------------------------------------- signals

def dense_pass(video, out, fps, width):
    """One ffmpeg pass: dense/fNNNNN.jpg at `fps` and `width`, plus per-sample signals from N px gray frames.
    Sample k is at k / fps; values at k describe the change from sample k-1 to k."""
    dd = os.path.join(out, "dense")
    shutil.rmtree(dd, ignore_errors=True)
    os.makedirs(dd)
    fc = f"[0:v]fps={fps},split=2[a][b];[a]scale={width}:-2[o1];[b]scale={N}:{N},format=gray[o2]"
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-filter_complex", fc, "-map", "[o1]", "-q:v", "5",
           os.path.join(dd, "f%05d.jpg"), "-map", "[o2]", "-f", "rawvideo", "-"]
    sig = {k: [] for k in ("energy", "coverage", "luma", "dluma", "dx", "dy", "zoom", "pc_peak", "agree", "sharpness")}
    prev = prev_s = None
    for g in az.stream(cmd, N * N):
        s = spectra(g)
        luma = sum(g) / (N * N)
        if prev is None:
            e = cov = dl = dx = dy = z = ag = 0.0
            pk = 1.0
        else:
            e = az.mad(prev, g)
            cov = sum(1 for a, b in zip(prev, g) if abs(a - b) > 10) / (N * N)
            dl = luma - sig["luma"][-1]
            dx, dy, z, pk, ag = global_motion(prev_s, s)
        for k, v in (("energy", e), ("coverage", cov), ("luma", luma), ("dluma", dl), ("dx", dx), ("dy", dy),
                     ("zoom", z), ("pc_peak", pk), ("agree", ag), ("sharpness", sharpness(g))):
            sig[k].append(round(v, 4))
        prev, prev_s = g, s
    return sig


def sharpness(g):
    """Mean absolute horizontal gradient of an N px gray frame; drops under motion blur and soft focus."""
    return sum(abs(g[i + 1] - g[i]) for i in range(len(g) - 1) if (i + 1) % N) / (N * (N - 1))


def runs(flags):
    """[(i0, i1)] inclusive runs of True."""
    out, s = [], None
    for i, f in enumerate(flags + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def near(t, onsets, tol=SYNC_S):
    return any(abs(t - o) <= tol for o in onsets)


def shot_motion(shot, sig, fps, onsets):
    """Motion events, holds and camera for one shot from the dense signals (pairs straddling a cut are skipped)."""
    ks = [k for k in range(1, len(sig["energy"])) if (k - 1) / fps >= shot["start"] - 1e-6 and k / fps < shot["end"] - 1e-6]
    res = {"events": [], "main_event": None, "holds": [], "hold_ratio": 0.0, "camera": ["static"],
           "energy_peak": 0.0, "energy_mean": 0.0, "samples": len(ks)}
    if not ks:
        return res
    e = [sig["energy"][k] for k in ks]
    peak = max(e)
    res["energy_peak"], res["energy_mean"] = round(peak, 2), round(statistics.mean(e), 2)
    # a fade changes every pixel, so it would dwarf the moves; the threshold comes from the non-fade samples
    # (a bright card sliding onto a dark frame raises the luma too, so a locked-on shift is never a fade)
    fading = [x > ACTIVE_MIN and abs(sig["dluma"][k]) >= 0.6 * x
              and not (sig["pc_peak"][k] >= 0.4 and math.hypot(sig["dx"][k], sig["dy"][k]) >= 0.5) for k, x in zip(ks, e)]
    fading = [f and ((i > 0 and fading[i - 1]) or (i + 1 < len(fading) and fading[i + 1])) for i, f in enumerate(fading)]
    thr = max(ACTIVE_MIN, ACTIVE_REL * max((x for x, f in zip(e, fading) if not f), default=0))
    act = [x > thr or f for x, f in zip(e, fading)]
    for a, b in runs([not x for x in act]):          # fill gaps of one or two samples inside a move
        if 0 < a and b < len(act) - 1 and b - a < 2:
            act[a:b + 1] = [True] * (b - a + 1)
    t = lambda k: round(k / fps, 3)
    for a, b in runs([not x for x in act]):
        if (b - a + 1) / fps >= HOLD_MIN_S:
            res["holds"].append([t(ks[a] - 1), t(ks[b])])
    res["hold_ratio"] = round(sum(h[1] - h[0] for h in res["holds"]) / max(shot["duration"], 1e-6), 2)
    spans = []
    for a, b in runs(act):                           # split a run where it switches between fading and moving
        s0 = a
        for i in range(a + 1, b + 1):
            if fading[i] != fading[i - 1]:
                spans.append((s0, i - 1))
                s0 = i
        spans.append((s0, b))
    for a, b in spans:
        lo, hi = max(0, a - 1), min(len(ks) - 1, b + 1)    # one quiet sample either side keeps the tails
        kk = ks[a:b + 1]
        good = [k for k in kk if sig["pc_peak"][k] >= 0.05]
        tx = sum(sig["dx"][k] for k in good) / N
        ty = sum(sig["dy"][k] for k in good) / N
        scale = math.prod(1 + sig["zoom"][k] for k in kk)
        moved = max(abs(tx), abs(ty)) >= 0.03 or abs(scale - 1) >= 0.03
        zoomed = abs(scale - 1) >= 0.03 and abs(math.log(scale)) >= max(abs(tx), abs(ty))
        if zoomed:
            direction = "in" if scale > 1 else "out"
        elif moved:
            direction = ("right" if tx > 0 else "left") if abs(tx) >= abs(ty) else ("down" if ty > 0 else "up")
        else:
            direction = "n/a"
        agree = statistics.median(sig["agree"][k] for k in kk)
        cov = statistics.median(sig["coverage"][k] for k in kk)
        # camera: the whole frame moves (quadrants agree and enough pixels change), or it scales
        kind = "fade" if fading[a] else "camera" if zoomed or (moved and agree >= 0.75 and cov >= 0.2) else "element"
        # speed curve: phase correlation when it locks on (it reads a card sliding in from off-screen correctly,
        # where the energy only grows as more of the card shows); otherwise the frame-difference energy
        rng = ks[lo:hi + 1]
        axis = "zoom" if zoomed else "dx" if direction in ("left", "right") else "dy"
        locked = direction != "n/a" and statistics.median(sig["pc_peak"][k] for k in kk) >= 0.4
        vel = [sig[axis][k] if sig["pc_peak"][k] >= 0.05 else 0.0 for k in rng] if direction != "n/a" else None
        speed = [abs(sig["zoom"][k]) if zoomed else math.hypot(sig["dx"][k], sig["dy"][k])
                 for k in rng if sig["pc_peak"][k] >= 0.05] if locked else e[lo:hi + 1]
        ez = classify_easing(speed, fps, vel)
        start, end = t(kk[0] - 1), t(kk[-1])
        pk = kk[max(range(len(kk)), key=lambda i: e[a + i])]
        res["events"].append({"start": start, "end": end, "duration": round(end - start, 3), "peak_t": t(pk),
                              "kind": kind, "direction": direction, "travel": [round(tx, 3), round(ty, 3)],
                              "scale": round(scale, 3), "agree": round(agree, 2), "coverage": round(cov, 2), "energy": round(sum(e[a:b + 1]), 1),
                              "speed_source": "phase_correlation" if locked else "energy", **ez,
                              "on_onset": near(start, onsets)})
    if res["events"]:
        # the biggest move; a fade only when nothing moves (a fade touches every pixel, so it always has the most energy)
        res["main_event"] = max(range(len(res["events"])), key=lambda i: (res["events"][i]["kind"] != "fade", res["events"][i]["energy"]))
    cam = []
    for ev in res["events"]:
        if ev["kind"] != "camera":
            continue
        if ev["direction"] in ("in", "out"):
            lab = f"zoom_{ev['direction']}"
        else:  # content moving left means the camera pans right
            lab = {"left": "pan_right", "right": "pan_left", "up": "tilt_down", "down": "tilt_up"}[ev["direction"]]
        if lab not in cam:
            cam.append(lab)
    res["camera"] = cam or ["static"]
    return res


# ---------------------------------------------------------------- transitions

def soft_transitions(sig, fps, taken):
    """Animated transitions the hard-cut detector misses (motion design rarely hard-cuts): a short burst where
    most of the frame changes, i.e. coverage peaks at >= 0.5 and averages under half that 0.3-0.5 s either side."""
    cov, e, out = sig["coverage"], sig["energy"], []
    for k in range(1, len(cov)):
        c = cov[k]
        if c < 0.5 or e[k] < 8 or c < max(cov[max(1, k - 2):k + 3]):
            continue
        side = [cov[j] for j in (k - 5, k - 4, k - 3, k + 3, k + 4, k + 5) if 1 <= j < len(cov)]
        t = round((k - 0.5) / fps, 3)
        if side and statistics.mean(side) < 0.5 * c and all(abs(t - x) >= 0.4 for x in taken + [o["t"] for o in out]):
            out.append({"t": t, "type": "soft_cut", "confidence": 0.6, "score": round(c, 3)})
    return out


def split_long(bounds, sig, fps, duration, max_len=4.0):
    """Split shots longer than max_len at their quietest sample in the middle half (type "continuous"), so every
    strip the vision pass sees covers at most max_len seconds with 8 frames."""
    add = []

    def split(a, b):
        ks = [k for k in range(1, len(sig["energy"])) if a + (b - a) * 0.25 <= k / fps <= a + (b - a) * 0.75]
        if b - a <= max_len or not ks:
            return
        t = round(min(ks, key=lambda k: sig["energy"][k]) / fps, 3)
        add.append({"t": t, "type": "continuous", "confidence": 1.0, "score": 0.0})
        split(a, t)
        split(t, b)
    edges = [0.0] + [b["t"] for b in bounds] + [duration]
    for a, b in zip(edges, edges[1:]):
        split(a, b)
    return add


def transition(video, b, i, out, info, onsets, half=0.5, wfps=30, cell=160, cols=10):
    """30 fps frames from t-half to t+half: a labelled sheet (transitions/cutNNN.jpg) and a computed type."""
    td = os.path.join(out, "transitions")
    os.makedirs(td, exist_ok=True)
    t0 = max(0.0, b["t"] - half)
    n = round(2 * half * wfps)
    rows = math.ceil(n / cols)
    sheet = os.path.join(td, f"cut{i:03d}.jpg")
    fc = (f"[0:v]fps={wfps},split=2[a][b];[a]scale={cell}:-2,tile={cols}x{rows}:padding=2:color=white[o1];"
          f"[b]scale={N}:{N},format=gray[o2]")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{2 * half:.3f}", "-i", video, "-filter_complex", fc,
           "-map", "[o1]", "-frames:v", "1", "-q:v", "4", "-y", sheet, "-map", "[o2]", "-f", "rawvideo", "-"]
    g = list(az.stream(cmd, N * N))
    c = min(len(g) - 1, max(1, round((b["t"] - t0) * wfps)))
    label_sheet(sheet, info, cell, cols, t0 - b["t"], wfps, c)
    d = [0.0] + [az.mad(x, y) for x, y in zip(g, g[1:])]
    sp = [spectra(x) for x in g]
    mot = [(0.0, 0.0, 0.0, 0.0)] + [global_motion(x, y) for x, y in zip(sp, sp[1:])]
    sh = [sharpness(x) for x in g]
    # the pair that straddles the cut is garbage for motion; it is the biggest jump next to the expected frame
    cut_pair = max(range(max(1, c - 2), min(len(g), c + 3)), key=lambda k: d[k], default=c)
    around = [k for k in range(max(1, c - 3), min(len(g), c + 4)) if k != cut_pair and mot[k][3] >= 0.05]
    second = lambda xs: sorted(xs)[-2] if len(xs) >= 2 else 0.0   # needs two pairs, so one bad pair cannot decide
    speed = second([math.hypot(mot[k][0], mot[k][1]) / N for k in around])
    zoom = second([abs(mot[k][2]) for k in around])
    # sharpness dip next to the cut, each side against its own calm frames (two shots rarely share a sharpness)
    ratio = lambda near_, calm: min(near_) / statistics.median(calm) if near_ and statistics.median(calm) > 1 else 1.0
    blur = min(ratio(sh[max(0, c - 3):c], sh[:5]), ratio(sh[c:c + 4], sh[-5:]))
    active = sum(1 for x in d if x > 0.3 * max(d)) if max(d) > 0 else 0
    det = b["type"]
    if det not in ("cut", "soft_cut"):
        computed = det
    elif zoom >= 0.02:
        computed = "zoom_through"
    elif speed >= 0.04:
        computed = "whip"
    elif blur < 0.6:
        computed = "blur"
    elif active >= 4:
        computed = "animated"     # a multi-frame change that is not a blend: wipe, push, morph (vision names it)
    else:
        computed = "cut" if det == "cut" else "animated"
    return {"index": i, "t": b["t"], "detected": det, "computed": computed, "confidence": b["confidence"],
            "metrics": {"peak_diff": round(max(d), 2), "changing_frames": active, "max_speed": round(speed, 3),
                        "max_zoom": round(zoom, 3), "blur_ratio": round(blur, 2),
                        "luma_min": round(min(sum(x) / len(x) for x in g), 1) if g else None},
            "sheet": os.path.relpath(sheet, out), "on_onset": near(b["t"], onsets)}


def label_sheet(path, info, cell, cols, t_first, wfps, cut_k):
    """Write the offset in ms on each tile and outline the first frame after the cut in red."""
    Image, ImageDraw = az.pil()
    if not Image or not os.path.exists(path):
        return
    im = Image.open(path).convert("RGB")
    ch = round(cell * (info["height"] or 9) / (info["width"] or 16) / 2) * 2
    d = ImageDraw.Draw(im)
    for k in range(math.ceil(im.width / (cell + 2)) * math.ceil(im.height / (ch + 2))):
        x, y = (k % cols) * (cell + 2), (k // cols) * (ch + 2)
        if y >= im.height:
            break
        d.rectangle((x, y, x + 44, y + 11), fill="black")
        d.text((x + 2, y), f"{round((t_first + k / wfps) * 1000):+d}", fill="white")
        if k == cut_k:
            d.rectangle((x, y, x + cell - 1, y + ch - 1), outline="red", width=2)
    im.save(path, quality=85)


# ---------------------------------------------------------------- strips

def shot_strip(shot, out, fps, n=8, cell=360, cols=4):
    """Up to `n` dense frames spread over the shot in a grid, each labelled with its shot-relative time."""
    Image, ImageDraw = az.pil()
    if not Image:
        return None
    ks = [k for k in range(math.ceil(shot["start"] * fps - 1e-6), math.floor(shot["end"] * fps - 1e-6) + 1)
          if os.path.exists(os.path.join(out, "dense", f"f{k + 1:05d}.jpg"))]
    if not ks:
        return None
    pick = sorted({ks[round(i * (len(ks) - 1) / max(1, n - 1))] for i in range(n)}) if len(ks) > n else ks
    ims = []
    for k in pick:
        im = Image.open(os.path.join(out, "dense", f"f{k + 1:05d}.jpg")).convert("RGB")
        ims.append((k, im.resize((cell, round(im.height * cell / im.width)))))
    h = max(im.height for _, im in ims)
    c = min(cols, len(ims))
    sheet = Image.new("RGB", (c * (cell + 4), math.ceil(len(ims) / c) * (h + 4)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (k, im) in enumerate(ims):
        x, y = (i % c) * (cell + 4), (i // c) * (h + 4)
        sheet.paste(im, (x, y))
        d.rectangle((x, y, x + 58, y + 12), fill="black")
        d.text((x + 3, y + 1), f"+{k / fps - shot['start']:.2f}s", fill="white")
    sd = os.path.join(out, "strips")
    os.makedirs(sd, exist_ok=True)
    path = os.path.join(sd, f"shot{shot['index']:03d}.jpg")
    sheet.save(path, quality=82)
    shot["dense_frames"] = [f"dense/f{ks[0] + 1:05d}.jpg", f"dense/f{ks[-1] + 1:05d}.jpg"]
    return os.path.relpath(path, out)


# ---------------------------------------------------------------- vision

MOTION_PROMPT = """You are a senior motion designer reverse-engineering a SaaS product marketing video so a team can
rebuild each shot in GSAP at the same level. For each shot you get:
- a strip: up to 8 frames spread across the shot, each labelled with its time from the shot start (read left to right, top to bottom);
- when a next shot exists, a transition sheet: 30 frames at 30 fps from 0.5 s before to 0.5 s after the cut, labelled in ms,
  with the first frame after the cut outlined in red;
- computed signals (below): motion events with timings and an easing guess from the frame-difference energy curve,
  global motion (pan, zoom), holds, the detected transition, and whether cuts and motion starts land on audio onsets.
Use the computed timings and easing unless the frames clearly contradict them; say so in "notes" when they do.
Return ONLY a JSON array, one object per shot, with exactly these keys:
{"shot": <int>,
 "role": "hook|problem|solution|demo|feature|social_proof|cta|brand|transition|other",
 "elements": ["<each distinct thing on screen: UI card, headline, device, cursor, avatar, logo, chart, background>"],
 "animations": [{"element": "<which element>", "property": "position|scale|opacity|blur|rotation|mask|clip|colour|3d|path|text|counter",
                 "direction": "left|right|up|down|in|out|cw|ccw|n/a", "from": "<start state>", "to": "<end state>",
                 "start_s": <seconds from shot start>, "duration_s": <seconds>, "stagger_s": <seconds between items or null>,
                 "easing": "linear|ease-out|ease-in|ease-in-out|overshoot|bounce|spring|step"}],
 "camera": {"move": "static|push_in|pull_out|pan|tilt|orbit|dolly|parallax|shake|rack_focus", "direction": "<or n/a>",
            "amount": "subtle|medium|strong|n/a", "easing": "<or n/a>"},
 "transition_out": {"type": "cut|match_cut|whip|zoom_through|mask_wipe|morph|push|slide|dissolve|fade|blur|light_leak|flash|glitch|none",
                    "duration_s": <seconds or 0>, "evidence": "<what the transition sheet shows>"},
 "typography": {"present": <bool>, "text": ["<verbatim on-screen text, at most 8 words each>"], "weight": "<e.g. bold, medium>",
                "size_class": "hero|headline|body|label|none", "kinetic": "none|word_by_word|char_cascade|mask_reveal|scale_pop|typewriter|highlight|counter|other"},
 "ui_presentation": "real_screen_recording|rebuilt_ui|device_mockup|3d_device|illustration|live_action|none",
 "background": {"type": "flat|gradient|mesh_gradient|glow|glass|grid|photo|video|3d", "colours": ["<main colours>"], "motion": "static|drift|parallax|n/a"},
 "effects": ["<e.g. glass morphism, soft drop shadow, glow, grain, depth of field, cursor click ripple, orbit path, light sweep>"],
 "sound_sync": "<which motion lands on an audio onset according to the signals, or none>",
 "notes": "<one sentence: how to rebuild this shot>"}
Be literal: do not invent text you cannot read or motion you cannot see."""


def compact_signals(s, tr):
    m = s["motion"]
    return {"shot": s["index"], "start": s["start"], "end": s["end"], "duration": s["duration"],
            "camera": m["camera"], "holds": m["holds"],
            "events": [{k: ev[k] for k in ("start", "end", "kind", "direction", "scale", "easing", "confidence", "pulses",
                                          "stagger_s", "on_onset")} for ev in m["events"]],
            "transition_out": {k: tr[k] for k in ("t", "detected", "computed", "on_onset")} if tr else None}


def vision_pass(backend, shots, transitions, out, batch=4):
    by_t = {t["index"]: t for t in transitions}
    step = 1 if backend == "ollama" else batch
    for i in range(0, len(shots), step):
        group = shots[i:i + step]
        imgs, sig = [], []
        for s in group:
            tr = by_t.get(s.get("transition_out"))
            if s.get("strip"):
                imgs.append((f"Shot {s['index']} strip ({az.ts(s['start'])}-{az.ts(s['end'])}, {s['duration']:.2f}s)", s["strip"]))
            if tr and tr["sheet"]:
                imgs.append((f"Shot {s['index']} to {s['index'] + 1} transition sheet (cut at {az.ts(tr['t'])})", tr["sheet"]))
            sig.append(compact_signals(s, tr))
        if not imgs:
            continue
        az.log(f"  vision: shots {group[0]['index']}-{group[-1]['index']} via {backend}")
        try:
            res = az.parse_json(az.ask(backend, MOTION_PROMPT + "\n\nCOMPUTED SIGNALS:\n" + json.dumps(sig), imgs, out))
            res = res if isinstance(res, list) else [res]
            by = {r.get("shot"): r for r in res if isinstance(r, dict)}
            for k, s in enumerate(group):
                s["vision"] = by.get(s["index"]) or (res[k] if k < len(res) and isinstance(res[k], dict) else None)
        except Exception as e:
            az.log(f"  vision batch failed: {e}")


# ---------------------------------------------------------------- summary + markdown

def count(xs):
    return dict(Counter(x for x in xs if x not in (None, "")).most_common())


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 3) if xs else None


def summarize(shots, transitions, duration):
    evs = [e for s in shots for e in s["motion"]["events"]]
    mains = [s["motion"]["events"][s["motion"]["main_event"]] for s in shots if s["motion"]["main_event"] is not None]
    out = {"easing_main_events": count(e["easing"] for e in mains), "easing_all_events": count(e["easing"] for e in evs),
           "event_kinds": count(e["kind"] for e in evs), "event_duration_median_s": med(e["duration"] for e in evs),
           "stagger_median_s": med(e["stagger_s"] for e in evs), "camera": count(c for s in shots for c in s["motion"]["camera"]),
           "transitions_computed": count(t["computed"] for t in transitions if t["computed"] != "continuous"),
           "hold_ratio": round(sum(h[1] - h[0] for s in shots for h in s["motion"]["holds"]) / duration, 2) if duration else 0}
    vs = [s["vision"] for s in shots if isinstance(s.get("vision"), dict)]
    if vs:
        anims = [a for v in vs for a in v.get("animations") or [] if isinstance(a, dict)]
        g = lambda v, k, f: (v.get(k) or {}).get(f) if isinstance(v.get(k), dict) else None
        out["vision"] = {
            "roles": [v.get("role") for v in vs], "transitions": count(g(v, "transition_out", "type") for v in vs),
            "camera": count(g(v, "camera", "move") for v in vs), "animation_properties": count(a.get("property") for a in anims),
            "animation_easing": count(a.get("easing") for a in anims), "animation_duration_median_s": med(a.get("duration_s") for a in anims),
            "animation_stagger_median_s": med(a.get("stagger_s") for a in anims), "ui_presentation": count(v.get("ui_presentation") for v in vs),
            "background": count(g(v, "background", "type") for v in vs), "kinetic_type": count(g(v, "typography", "kinetic") for v in vs),
            "effects": count(x for v in vs for x in v.get("effects") or [] if isinstance(x, str))}
    return out


def markdown(m):
    src, p, su, au = m["source"], m["pacing"], m["summary"], m["audio"]
    meta = src.get("metadata") or {}
    fmt = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "none"
    L = [f"# Motion breakdown: {meta.get('title') or src['slug']}", ""]
    if meta:
        L += [f"- Source: {meta.get('webpage_url', '')}", f"- Creator: {meta.get('uploader') or meta.get('channel') or 'n/a'}"]
    L += [f"- Duration: {src['duration_s']}s, {src['width']}x{src['height']} @ {src['fps']} fps; dense frames at "
          f"{m['settings']['dense_fps']} fps; vision: {m['vision']['status']} ({m['vision']['backend']})",
          f"- {p['shot_count']} shots, {p['cuts_per_10s']} cuts per 10 s, average shot {p['avg_shot_s']}s", "",
          "## Motion stats (computed)", "",
          f"- Easing of each shot's main move: {fmt(su['easing_main_events'])}",
          f"- Easing of all moves: {fmt(su['easing_all_events'])}",
          f"- Move kinds: {fmt(su['event_kinds'])}; median move {su['event_duration_median_s']}s; median stagger {su['stagger_median_s']}s",
          f"- Camera: {fmt(su['camera'])}; holds cover {int(su['hold_ratio'] * 100)}% of the runtime",
          f"- Transitions: {fmt(su['transitions_computed'])}"]
    if au.get("status") == "ok":
        L.append(f"- Sound sync: {au['cuts_on_onset']}/{p['cut_count']} cuts and {au['events_on_onset']}/{au['event_count']} "
                 f"move starts land within {int(SYNC_S * 1000)} ms of an audio onset (chance about {int(au['chance_ratio'] * 100)}%)")
    v = su.get("vision")
    if v:
        L += ["", "## Motion stats (vision)", "",
              f"- Roles in order: {' > '.join(str(r) for r in v['roles'])}",
              f"- Transitions: {fmt(v['transitions'])}", f"- Camera: {fmt(v['camera'])}",
              f"- Animated properties: {fmt(v['animation_properties'])}",
              f"- Easing: {fmt(v['animation_easing'])}; median duration {v['animation_duration_median_s']}s, median stagger {v['animation_stagger_median_s']}s",
              f"- UI presentation: {fmt(v['ui_presentation'])}", f"- Background: {fmt(v['background'])}",
              f"- Kinetic type: {fmt(v['kinetic_type'])}", f"- Effects: {fmt(dict(list(v['effects'].items())[:12]))}"]
    tr = {t["index"]: t for t in m["transitions"]}
    L += ["", "## Shots", ""]
    for s in m["shots"]:
        mo, vi = s["motion"], s.get("vision") or {}
        L += [f"### Shot {s['index']}: {az.ts(s['start'])}-{az.ts(s['end'])} ({s['duration']:.2f}s)"
              + (f", {vi.get('role')}" if vi.get("role") else ""), ""]
        L.append(f"- Computed: camera {', '.join(mo['camera'])}; holds {mo['holds'] or 'none'}")
        for ev in mo["events"]:
            L.append(f"  - {ev['start']:.2f}-{ev['end']:.2f}s {ev['kind']} {ev['direction']}: {ev['easing']} "
                     f"({ev['confidence']}), scale {ev['scale']}, travel {ev['travel']}"
                     + (f", {ev['pulses']} pulses, stagger {ev['stagger_s']}s" if ev["stagger_s"] else "")
                     + (", on audio onset" if ev["on_onset"] else ""))
        if vi:
            L.append(f"- Elements: {', '.join(map(str, vi.get('elements') or []))}")
            an = [a for a in vi.get("animations") or [] if isinstance(a, dict)]
            if an:
                L += ["", "| Element | Property | From | To | Start | Dur | Ease | Stagger |", "|---|---|---|---|---|---|---|---|"]
                for a in an:
                    L.append("| " + " | ".join(str(a.get(k, "")).replace("|", "/") for k in
                                                ("element", "property", "from", "to", "start_s", "duration_s", "easing", "stagger_s")) + " |")
                L.append("")
            cam, ty, bg = vi.get("camera") or {}, vi.get("typography") or {}, vi.get("background") or {}
            if isinstance(cam, dict):
                L.append(f"- Camera: {cam.get('move')} {cam.get('direction', '')} {cam.get('amount', '')} {cam.get('easing', '')}".rstrip())
            if isinstance(ty, dict) and ty.get("present"):
                L.append(f"- Type: {ty.get('weight')} {ty.get('size_class')}, kinetic {ty.get('kinetic')}: "
                         f"{' / '.join(map(str, ty.get('text') or []))}")
            L.append(f"- UI: {vi.get('ui_presentation')}; background: {bg.get('type') if isinstance(bg, dict) else bg}"
                     f" {', '.join(map(str, bg.get('colours') or [])) if isinstance(bg, dict) else ''}")
            if vi.get("effects"):
                L.append(f"- Effects: {', '.join(map(str, vi['effects']))}")
            if vi.get("sound_sync"):
                L.append(f"- Sound: {vi['sound_sync']}")
            if vi.get("notes"):
                L.append(f"- Rebuild: {vi['notes']}")
        t = tr.get(s.get("transition_out"))
        if t and t["computed"] == "continuous":
            L.append(f"- Continues into the next segment at {az.ts(t['t'])} (long shot split for analysis, no cut)")
        elif t:
            vt = (vi.get("transition_out") or {}) if isinstance(vi.get("transition_out"), dict) else {}
            L.append(f"- Transition out at {az.ts(t['t'])}: computed {t['computed']}"
                     + (f", vision {vt.get('type')} ({vt.get('duration_s')}s): {vt.get('evidence', '')}" if vt else "")
                     + (", on audio onset" if t["on_onset"] else "") + f". Sheet: `{t['sheet']}`")
        if s.get("strip"):
            L.append(f"- Strip: `{s['strip']}`")
        L.append("")
    L += ["Full data: `motion.json`. Dense frames: `dense/`.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def analyze_motion(video, out_dir=".", vision="auto", scene_threshold=0.3, dense_fps=10, dense_width=480,
                   vision_batch=4, meta_path=None):
    """Write <out_dir>/<slug>/motion.json + motion.md, dense/, strips/, transitions/. Returns that folder."""
    video = os.path.abspath(video)
    if not os.path.exists(video):
        sys.exit(f"Video not found: {video}")
    meta_path = meta_path or os.path.splitext(video)[0] + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    slug = az.slugify(os.path.splitext(os.path.basename(video))[0])
    out = os.path.join(os.path.abspath(out_dir), slug)
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    az.log(f"Motion analysis of {video}\n  -> {out}")
    info = az.probe(video)
    m = {"schema_version": SCHEMA_VERSION, "mode": "motion", "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "source": {"video_path": video, "slug": slug, **info, "metadata": meta},
         "settings": {"dense_fps": dense_fps, "dense_width": dense_width, "signal_size": N, "window_s": 0.5, "window_fps": 30,
                      "scene_threshold": scene_threshold}}

    az.log("[1/5] shots")
    bounds = az.detect_boundaries(az.frame_stats(video, tmp), info["fps"] or 30, scene_threshold, g=az.thumbs(video))

    az.log("[2/5] audio onsets")
    onsets = []
    if info["has_audio"]:
        onsets = [round(i * az.HOP / az.SR, 3) for i, _, _ in az.onsets(az.envelope_db(video))]

    az.log(f"[3/5] dense frames at {dense_fps} fps + motion signals")
    sig = dense_pass(video, out, dense_fps, dense_width)
    m["signals"] = {"fps": dense_fps, "note": "sample k is at k/fps; values describe the change from sample k-1",
                    **{k: v for k, v in sig.items() if k != "sharpness"}}
    bounds = sorted(bounds + soft_transitions(sig, dense_fps, [b["t"] for b in bounds]), key=lambda b: b["t"])
    m["pacing"] = az.pacing(az.build_shots(bounds, info["duration_s"]), info["duration_s"])
    bounds = sorted(bounds + split_long(bounds, sig, dense_fps, info["duration_s"]), key=lambda b: b["t"])
    shots = az.build_shots(bounds, info["duration_s"])
    az.log(f"  {m['pacing']['shot_count']} shots ({m['pacing']['cut_count']} cuts incl. soft), {len(shots)} segments")
    for i, s in enumerate(shots):
        s["motion"] = shot_motion(s, sig, dense_fps, onsets)
        s["strip"] = shot_strip(s, out, dense_fps)
        s["transition_out"] = i + 1 if i < len(bounds) else None

    az.log(f"[4/5] transitions ({len(bounds)} cuts, 30 fps windows)")
    shutil.rmtree(os.path.join(out, "transitions"), ignore_errors=True)
    m["transitions"] = [transition(video, b, i + 1, out, info, onsets) if b["type"] != "continuous" else
                        {"index": i + 1, "t": b["t"], "detected": "continuous", "computed": "continuous", "confidence": 1.0,
                         "metrics": {}, "sheet": None, "on_onset": near(b["t"], onsets)} for i, b in enumerate(bounds)]
    evs = [e for s in shots for e in s["motion"]["events"]]
    dur = info["duration_s"] or 1
    m["audio"] = {"status": "ok", "onsets": onsets, "onsets_per_s": round(len(onsets) / dur, 2),
                  "cuts_on_onset": sum(t["on_onset"] for t in m["transitions"] if t["sheet"]),
                  "events_on_onset": sum(e["on_onset"] for e in evs), "event_count": len(evs),
                  "chance_ratio": round(min(1.0, len(onsets) / dur * 2 * SYNC_S), 2)} if info["has_audio"] else {"status": "no_audio"}

    az.log("[5/5] vision")
    vb = az.pick_backend(vision)
    m["vision"] = {"backend": vb, "status": "skipped" if vb == "none" else "ok"}
    if vb == "none":
        m["vision"]["todo"] = "pass --vision claude-cli (your Claude Code login), or set ANTHROPIC_API_KEY, or run an ollama vision model"
    else:
        vision_pass(vb, shots, m["transitions"], out, vision_batch)
        if not any(s.get("vision") for s in shots):
            m["vision"]["status"] = "failed"
    m["shots"] = shots
    m["summary"] = summarize(shots, m["transitions"], info["duration_s"])
    json.dump(m, open(os.path.join(out, "motion.json"), "w"), indent=1)
    open(os.path.join(out, "motion.md"), "w").write(markdown(m))
    shutil.rmtree(tmp, ignore_errors=True)
    az.log(f"Done: {os.path.join(out, 'motion.md')}")
    return out
