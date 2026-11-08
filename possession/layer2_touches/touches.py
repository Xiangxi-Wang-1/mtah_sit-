"""Touches and control segments from the cleaned ball and the layer-1 features.

Two kinds of evidence, merged into one event list:

  control segments  the ball is with a player: his distance to the ball is under CONTROL_D and
                    he moves with it (relative speed under CONTROL_RELV) for CONTROL_MIN_FRAMES.
                    Its start is a reception, its end a release.
  free touches      outside control segments, the ball's own motion changes: a local minimum of
                    speed (it was stopped or redirected) or a sharp turn of heading. Each is
                    attributed to the nearest player within TOUCH_ATTRIB_WINDOW frames, if one
                    is within TOUCH_ATTRIB_RADIUS.

Also marks where the ball is lost and found (gap edges), because a hand-over can hide in a gap.
"""

import numpy as np
from scipy.signal import find_peaks

from .. import config as C

# event kinds
RECEIVE, RELEASE, TOUCH, UNATTRIBUTED, LOST, FOUND, ARTIFACT, STOP, KICK = range(9)
KIND_NAMES = ["receive", "release", "touch", "unattributed", "lost", "found", "artifact", "stop",
              "kick"]


def _runs(mask):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def heading_change(vel, k):
    """Angle (deg) between the ball's velocity k frames before and k frames after each frame."""
    out = np.full(len(vel), np.nan)
    u, w = vel[2 * k:], vel[:-2 * k]
    with np.errstate(all="ignore"):
        cos = np.sum(u * w, axis=1) / (np.hypot(*u.T) * np.hypot(*w.T))
    out[k:-k] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out


def control_segments(F, half):
    """[(start, end, team, slot)] of the ball being with one player."""
    d = np.fmin(F["d_h"], F["d_a"])
    home_nearer = F["d_h"] <= F["d_a"]
    relv = np.where(home_nearer, F["relv_h"], F["relv_a"])
    slot = np.where(home_nearer, F["slot_h"], F["slot_a"]).astype(int)
    team = np.where(home_nearer, 0, 1)
    ok = (d < C.CONTROL_D) & (relv < C.CONTROL_RELV) & (half > 0)
    ok = np.nan_to_num(ok, nan=0).astype(bool)
    a, b = _runs(ok)
    segs = []
    for s, e in zip(a, b):
        # one segment per player: split where the nearest player changes
        sl = slot[s:e]
        cuts = np.flatnonzero(np.diff(sl) != 0) + 1
        for x, y in zip(np.r_[0, cuts], np.r_[cuts, len(sl)]):
            segs.append([s + x, s + y, int(team[s + x]), int(sl[x])])
    # merge same-player segments separated by a short break
    merged = []
    for seg in segs:
        if merged and seg[3] == merged[-1][3] and seg[2] == merged[-1][2] \
                and seg[0] - merged[-1][1] <= C.CONTROL_MAX_BREAK:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    out = []
    for s, e, tm, sl in merged:
        if e - s < C.CONTROL_MIN_FRAMES:
            continue
        opp = F["d_a"][s:e] if tm == 0 else F["d_h"][s:e]
        contest = float(np.mean(np.nan_to_num(opp, nan=99) < C.CONTROL_D))
        out.append((s, e, tm, sl, contest))
    return out


def kinematic_candidates(speed, vel, half):
    """Frames where the ball's own motion changes, by kind.

    Returns (frames, kinds, turn) where kinds is TOUCH-candidate (a credible redirect or a
    moderate speed dip), ARTIFACT (a near-reversal at speed - the tracker, not a player) or STOP
    (a very deep speed dip - the ball coming to rest).
    """
    x = np.where(np.isnan(speed), 99.0, speed)   # a gap is not a minimum
    mins, props = find_peaks(-x, prominence=C.TOUCH_SPEED_MIN_PROMINENCE)
    prom = props["prominences"]
    turn = heading_change(vel, C.TOUCH_TURN_K)
    moving = speed > C.TOUCH_TURN_MIN_SPEED
    turns, _ = find_peaks(np.nan_to_num(np.where(moving, turn, 0)), height=C.TOUCH_TURN_DEG,
                          distance=C.TOUCH_MERGE)
    lo, hi = C.TOUCH_TURN_SPEED_RANGE
    kinds = {}
    for f in turns:
        if turn[f] > C.TOUCH_TURN_MAX_DEG and speed[f] >= 6.0:
            kinds[f] = ARTIFACT
        elif lo <= speed[f] <= hi or turn[f] <= C.TOUCH_TURN_MAX_DEG:
            kinds[f] = TOUCH
        else:
            kinds[f] = ARTIFACT
    for f, p in zip(mins, prom):
        k = STOP if p >= C.TOUCH_SPEED_MIN_MAX_PROMINENCE else TOUCH
        # a turn and a dip at the same moment are one event; a credible touch wins
        kinds[f] = min(kinds.get(f, k), k) if k == TOUCH else kinds.get(f, k)
    frames = np.array(sorted(f for f in kinds if half[f] > 0), dtype=int)
    merged, mk = [], []
    for f in frames:
        if merged and f - merged[-1] < C.TOUCH_MERGE:
            if kinds[f] == TOUCH:
                mk[-1] = TOUCH
            continue
        merged.append(int(f))
        mk.append(kinds[f])
    return np.array(merged, dtype=int), np.array(mk, dtype=int), turn


def kicks(speed, F, half, in_ctrl):
    """The ball accelerating away from a player: (frame, team, slot, dist, speed before, after).

    The start of a rise of >= KICK_RISE m/s within KICK_WINDOW frames, with a player within
    KICK_RADIUS at that moment. Kicks at the end of a control segment are releases already and
    are skipped."""
    s = np.nan_to_num(speed, nan=-1.0)
    n = len(s)
    out = []
    f = 1
    while f < n - C.KICK_WINDOW:
        if half[f] == 0 or s[f] < 0 or s[f] > s[f - 1]:
            f += 1
            continue
        # f is a local start of a rise: the speed does not fall from f to f+1
        ahead = s[f:f + C.KICK_WINDOW]
        if (ahead < 0).any() or ahead.max() - s[f] < C.KICK_RISE or s[f + 1] < s[f]:
            f += 1
            continue
        lo, hi = max(0, f - 3), min(n, f + 3)
        dh, da = F["d_h"][lo:hi], F["d_a"][lo:hi]
        both = np.fmin(dh, da)
        if np.isnan(both).all() or np.nanmin(both) > C.KICK_RADIUS:
            f += C.KICK_WINDOW
            continue
        k = int(np.nanargmin(both))
        team = 0 if (dh[k] <= da[k] or np.isnan(da[k])) else 1
        slot = int(F["slot_h"][lo + k] if team == 0 else F["slot_a"][lo + k])
        near_ctrl_end = in_ctrl[max(0, f - 6):f + 1].any()
        if not near_ctrl_end:
            out.append((f, team, slot, float(both[k]), float(s[f]), float(ahead.max())))
        f += C.KICK_WINDOW
    return out


def attribute(frames, F):
    """Nearest player (team, slot, distance) within +/- TOUCH_ATTRIB_WINDOW of each frame."""
    out = []
    n = len(F["d_h"])
    for f in frames:
        lo, hi = max(0, f - C.TOUCH_ATTRIB_WINDOW), min(n, f + C.TOUCH_ATTRIB_WINDOW + 1)
        dh, da = F["d_h"][lo:hi], F["d_a"][lo:hi]
        both = np.fmin(dh, da)
        if np.isnan(both).all():
            out.append((-1, -1, np.nan, f))
            continue
        k = int(np.nanargmin(both))
        if dh[k] <= da[k] or np.isnan(da[k]):
            out.append((0, int(F["slot_h"][lo + k]), float(dh[k]), lo + k))
        else:
            out.append((1, int(F["slot_a"][lo + k]), float(da[k]), lo + k))
    return out


def detect(z, F):
    """Event table and per-frame arrays."""
    half = z["half"].astype(int)
    speed = z["ball_speed"].astype(float)
    vel = z["ball_vel"].astype(float)
    n = len(half)

    segs = control_segments(F, half)
    in_ctrl = np.zeros(n, bool)
    ctrl_team = np.full(n, -1, np.int8)
    for s, e, t, sl, _ in segs:
        in_ctrl[s:e] = True
        ctrl_team[s:e] = t

    cand, ckind, turn = kinematic_candidates(speed, vel, half)
    # free events: kinematic events not inside (or right at the edge of) a control segment
    near_ctrl = np.convolve(in_ctrl, np.ones(2 * C.TOUCH_MERGE + 1), "same") > 0
    keep = ~near_ctrl[cand]
    free, fkind = cand[keep], ckind[keep]
    att = attribute(free, F)

    events = []   # (frame, kind, team, slot, dist, speed_before, speed_after, turn, duration,
    #             contest)

    def sp(a, b):
        with np.errstate(all="ignore"):
            return float(np.nanmean(speed[max(0, a):max(0, b)])) if b > a else np.nan

    for s, e, t, sl, contest in segs:
        events.append((s, RECEIVE, t, sl, float(np.fmin(F["d_h"][s], F["d_a"][s])),
                       sp(s - 8, s), sp(s, s + 8), np.nan, e - s, contest))
        events.append((e - 1, RELEASE, t, sl, float(np.fmin(F["d_h"][e - 1], F["d_a"][e - 1])),
                       sp(e - 8, e), sp(e, e + 8), np.nan, e - s, contest))
    for f, ck, (t, sl, d, at) in zip(free, fkind, att):
        if ck != TOUCH:
            events.append((int(f), int(ck), -1, -1, d, sp(f - 8, f), sp(f, f + 8),
                           float(turn[f]), 0, np.nan))
            continue
        kind = TOUCH if (t >= 0 and d <= C.TOUCH_ATTRIB_RADIUS) else UNATTRIBUTED
        events.append((int(f), kind, t if kind == TOUCH else -1, sl if kind == TOUCH else -1,
                       d, sp(f - 8, f), sp(f, f + 8), float(turn[f]), 0, np.nan))
    for f, t, sl, d, s0, s1 in kicks(speed, F, half, in_ctrl):
        events.append((f, KICK, t, sl, d, s0, s1, np.nan, 0, np.nan))
    gl = z["ball_gap_len"]
    known = gl == 0
    for f in np.flatnonzero(known[:-1] & ~known[1:] & (half[:-1] > 0)):
        events.append((int(f), LOST, -1, -1, np.nan, sp(f - 8, f), np.nan, np.nan, 0, np.nan))
    for f in np.flatnonzero(~known[:-1] & known[1:] & (half[1:] > 0)) + 1:
        events.append((int(f), FOUND, -1, -1, np.nan, np.nan, sp(f, f + 8), np.nan, 0, np.nan))
    events.sort(key=lambda r: (r[0], r[1]))
    E = np.array(events, dtype=[("frame", "i4"), ("kind", "i1"), ("team", "i1"), ("slot", "i1"),
                                ("dist", "f4"), ("speed_before", "f4"), ("speed_after", "f4"),
                                ("turn", "f4"), ("duration", "i4"), ("contest", "f4")])

    # per-frame ball state: 0 unknown, 1 with a player, 2 travelling
    state = np.where(known, 2, 0).astype(np.int8)
    state[in_ctrl] = 1
    # the most recent control, carried forward: who last had the ball, how long ago, and
    # whether that control was contested
    last_team = np.full(n, -1, np.int8)
    since = np.full(n, -1, np.int32)
    last_contest = np.full(n, np.nan, np.float32)
    starts = sorted(segs)
    for k, (s, e, t, sl, contest) in enumerate(starts):
        stop = starts[k + 1][0] if k + 1 < len(starts) else n
        last_team[s:stop] = t
        since[s:stop] = np.arange(stop - s)
        last_contest[s:stop] = contest
    return E, {"ball_state": state, "control_team": ctrl_team, "turn": turn.astype(np.float32),
               "last_control_team": last_team, "frames_since_control": since,
               "last_control_contest": last_contest}
