"""Clean the ball track: drop false detections, fill short gaps, smooth, differentiate.

All positions in metres. A frame's ball is in one of four states (BALL_* codes below):
observed and kept, filled by interpolation, removed as a false detection, or missing.
"""

import numpy as np
from scipy.signal import savgol_filter

from .. import config as C

MISSING, FILLED, OBSERVED, REMOVED = 0, 1, 2, 3


def _consistent(p, q, gap_frames, v_max=C.BALL_V_MAX, jitter=C.BALL_JITTER):
    """Could one ball be at p and, gap_frames later, at q?"""
    return np.hypot(*(q - p)) <= v_max * C.DT * gap_frames + jitter


def _tracklets(xy, visible, half):
    """Split visible frames into tracklets: chains of physically consistent detections.

    Two successive detections are chained when they are in the same half, at most
    BALL_MAX_FILL_GAP + 1 frames apart, and reachable at BALL_V_MAX. Returns a list of
    index arrays.
    """
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        return []
    gaps = np.diff(idx)
    step = np.hypot(*(xy[idx[1:]] - xy[idx[:-1]]).T)
    linked = ((gaps <= C.BALL_MAX_FILL_GAP + 1)
              & (half[idx[1:]] == half[idx[:-1]])
              & (step <= C.BALL_V_MAX * C.DT * gaps + C.BALL_JITTER))
    cuts = np.flatnonzero(~linked) + 1
    return np.split(idx, cuts)


def remove_overspeed(xy):
    """Mask of frames reached by a step faster than BALL_V_MAX.

    The provider's ball track is already smoothed, so when the tracker re-locks onto another
    object it does not jump: it glides there at 45-75 m/s over 5-20 frames. No ball moves that
    fast, so those frames are not the ball. Both ends of each over-speed step are dropped.
    """
    step = np.hypot(*(xy[1:] - xy[:-1]).T) / C.DT
    fast = np.zeros(len(xy), dtype=bool)
    hit = np.flatnonzero(step > C.BALL_V_MAX)  # NaN compares False
    fast[hit] = True
    fast[hit + 1] = True
    return fast


def remove_false_detections(xy, half, passes=3):
    """Mask of detections to drop.

    A short tracklet (< BALL_MIN_TRACKLET frames) that cannot be joined to the tracklet before
    it or after it is a false detection: a one-frame spike onto a player's head or a line
    marking. Repeated, because removing a spike lets its two neighbours chain up again.
    """
    visible = ~np.isnan(xy[:, 0])
    removed = np.zeros(len(xy), dtype=bool)
    for _ in range(passes):
        tracks = _tracklets(xy, visible & ~removed, half)
        changed = False
        for k, t in enumerate(tracks):
            if len(t) >= C.BALL_MIN_TRACKLET:
                continue
            joins = False
            for other, mine in ((tracks[k - 1][-1] if k > 0 else None, t[0]),
                                (tracks[k + 1][0] if k + 1 < len(tracks) else None, t[-1])):
                if other is None or half[other] != half[mine]:
                    continue
                gap = abs(mine - other)
                if _consistent(xy[other], xy[mine], gap):
                    joins = True
            if not joins:
                # Isolated short detections with nothing either side within reach are kept
                # only if they are long enough to stand on their own.
                neighbours = [tracks[j] for j in (k - 1, k + 1) if 0 <= j < len(tracks)]
                near = any(abs(n[-1] - t[0]) <= 2 * C.FPS or abs(n[0] - t[-1]) <= 2 * C.FPS
                           for n in neighbours)
                if near:
                    removed[t] = True
                    changed = True
        if not changed:
            break
    return removed


def _endpoint_velocity(xy, idx_run, at_end):
    """Velocity at the start or end of a run of detections, from up to 3 samples."""
    take = idx_run[-3:] if at_end else idx_run[:3]
    if len(take) < 2:
        return np.zeros(2)
    t = take.astype(float) * C.DT
    return np.array([np.polyfit(t, xy[take, d], 1)[0] for d in range(2)])


def fill_gaps(xy, half, method="linear", max_gap=C.BALL_MAX_FILL_GAP):
    """Interpolate gaps of <= max_gap frames between consistent detections.

    method: "linear" (straight line, constant speed) or "hermite" (cubic matching the velocity
    at both ends; follows curving or decelerating balls but overshoots at a touch).
    Returns (filled_xy, filled_mask).
    """
    out = xy.copy()
    filled = np.zeros(len(xy), dtype=bool)
    visible = ~np.isnan(xy[:, 0])
    idx = np.flatnonzero(visible)
    if idx.size < 2:
        return out, filled
    gaps = np.diff(idx)
    for g in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[g], idx[g + 1]
        if half[a] != half[b] or half[a] == 0:
            continue
        if not _consistent(xy[a], xy[b], b - a):
            continue
        frames = np.arange(a + 1, b)
        s = (frames - a) / float(b - a)
        if method == "hermite":
            run_a = idx[max(0, g - 2):g + 1]
            run_b = idx[g + 1:g + 4]
            # Only trust endpoint velocities estimated from consecutive frames.
            run_a = run_a[run_a >= a - 2]
            run_b = run_b[run_b <= b + 2]
            span = (b - a) * C.DT
            va = _endpoint_velocity(xy, run_a, True) * span
            vb = _endpoint_velocity(xy, run_b, False) * span
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            out[frames] = (h00[:, None] * xy[a] + h10[:, None] * va
                           + h01[:, None] * xy[b] + h11[:, None] * vb)
        else:
            out[frames] = xy[a] + s[:, None] * (xy[b] - xy[a])
        filled[frames] = True
    return out, filled


def _runs(mask):
    """(start, stop) of each run of True in mask."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def smooth_and_differentiate(xy, window=C.BALL_SG_WINDOW, order=C.BALL_SG_ORDER):
    """Savitzky-Golay smoothed position and velocity within each continuous run.

    Runs shorter than the window keep their raw position and get a finite-difference velocity;
    single-frame runs get no velocity (NaN).
    """
    pos = np.full_like(xy, np.nan)
    vel = np.full_like(xy, np.nan)
    for a, b in _runs(~np.isnan(xy[:, 0])):
        seg = xy[a:b]
        n = b - a
        if n >= window:
            pos[a:b] = savgol_filter(seg, window, order, axis=0, mode="interp")
            vel[a:b] = savgol_filter(seg, window, order, deriv=1, delta=C.DT, axis=0,
                                     mode="interp")
        else:
            pos[a:b] = seg
            if n >= 2:
                vel[a:b] = np.gradient(seg, C.DT, axis=0)
    return pos, vel


def clean_ball(xy_raw, half, fill_method=C.BALL_FILL_METHOD):
    """The whole ball step. Returns dict with xy, vel, speed, state."""
    fast = remove_overspeed(xy_raw)
    xy = xy_raw.copy()
    xy[fast] = np.nan
    removed = remove_false_detections(xy, half) | fast
    xy[removed] = np.nan
    xy, filled = fill_gaps(xy, half, method=fill_method)
    pos, vel = smooth_and_differentiate(xy)

    state = np.full(len(xy_raw), MISSING, dtype=np.int8)
    state[~np.isnan(xy_raw[:, 0])] = OBSERVED
    state[removed] = REMOVED
    state[filled] = FILLED
    out = {"xy": pos, "vel": vel, "speed": np.hypot(vel[:, 0], vel[:, 1]), "state": state,
           "removed": removed, "overspeed": fast}
    out.update(gap_context(pos))
    return out


def gap_context(xy):
    """Where the ball was last seen and will next be seen, for every frame.

    Most missing ball is in long blocks (5-80 s) that start where the ball leaves play and end
    on a restart, so the edges of a gap carry information later layers need.
    """
    n = len(xy)
    known = ~np.isnan(xy[:, 0])
    idx = np.arange(n)
    last = np.where(known, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(known, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, -1)
    until = np.where(nxt < n, nxt - idx, -1)
    gap_len = np.where(known, 0, np.where((last >= 0) & (nxt < n), nxt - last - 1, -1))
    last_xy = np.where((last >= 0)[:, None], xy[np.clip(last, 0, n - 1)], np.nan)
    next_xy = np.where((nxt < n)[:, None], xy[np.clip(nxt, 0, n - 1)], np.nan)
    return {"frames_since_seen": since.astype(np.int32), "frames_until_seen": until.astype(np.int32),
            "gap_len": gap_len.astype(np.int32), "last_seen_xy": last_xy, "next_seen_xy": next_xy}
