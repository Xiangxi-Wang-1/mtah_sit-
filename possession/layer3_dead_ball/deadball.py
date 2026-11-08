"""Dead-ball detection: per-frame evidence, two-state Viterbi, boundary snapping.

The ball is dead from the moment play stops (ball over a line, or the whistle) to the restart
kick. What the tracking shows during a stoppage:

  - the ball beyond a line, or the ball lost - usually for 5-80 s, and last seen near a line
  - everybody slowing down (median player speed halves)
  - the ball at rest, often on a restart spot, then kicked
  - a Smart Tagging set-piece window around the restart

Each cue adds log-odds of "dead" per frame (config DEAD_*). A two-state Viterbi with a switch
cost turns that into stretches, so one slow moment does not stop play. Each boundary is then
moved to the physical event that marks it: a restart to the kick (ball speed rising from rest,
or the ball re-appearing already moving), a stoppage to the ball crossing a line or being lost.
"""

import numpy as np

from .. import config as C
from ..common import restart_spots
from ..layer1_features.features import rolling_mean


def evidence(z, F, L2):
    """Per-frame log-odds that the ball is dead, and the breakdown by cue."""
    n = len(z["half"])
    gl = z["ball_gap_len"].astype(float)
    known = gl == 0
    out = np.nan_to_num(F["ball_outside"].astype(float))
    speed = F["ball_speed"].astype(float)
    ms3 = rolling_mean(F["mean_speed_all"].astype(float), 37)
    parts = {}
    parts["outside"] = np.where(out > C.DEAD_OUTSIDE_M, C.DEAD_W_OUTSIDE, 0.0)
    gap = np.where(gl >= C.DEAD_GAP_LONG, C.DEAD_W_GAP_LONG,
                   np.where(gl >= C.DEAD_GAP_MID, C.DEAD_W_GAP_MID, 0.0))
    edge = (F["last_seen_to_line"].astype(float) < C.DEAD_EDGE_M) | \
           (F["last_seen_outside"].astype(float) > 0)
    parts["gap"] = gap + np.where((gl >= C.DEAD_GAP_MID) & edge, C.DEAD_W_GAP_EDGE, 0.0)
    parts["players_slow"] = np.nan_to_num(C.DEAD_W_SPEED * (C.DEAD_SPEED_PIVOT - ms3))
    inside = known & (out == 0)
    parts["ball_fast"] = np.where(inside & (np.nan_to_num(speed) > C.DEAD_BALL_FAST),
                                  C.DEAD_W_BALL_FAST, 0.0)
    far = np.nan_to_num(F["ball_to_line"].astype(float)) > 5
    parts["control"] = np.where((L2["ball_state"] == 1) & far, C.DEAD_W_CONTROL, 0.0)
    slow = known & (np.nan_to_num(speed, nan=99) < C.DEAD_STILL_SPEED)
    idx = np.arange(n)
    last_fast = np.maximum.accumulate(np.where(~slow, idx, -1))
    still_for = np.where(slow, idx - last_fast, 0)
    still = still_for >= C.DEAD_STILL_FRAMES
    spot = np.nan_to_num(F["ball_to_restart_spot"].astype(float), nan=99) < C.DEAD_SPOT_M
    parts["ball_still"] = np.where(still, C.DEAD_W_STILL, 0.0) + \
        np.where(still & spot, C.DEAD_W_SPOT, 0.0)
    sp = F["st_set_piece"] > 0
    # first half of each set-piece window: the restart is inside it, so before it is dead
    first_half = np.zeros(n, bool)
    a = np.flatnonzero(sp[1:] & ~sp[:-1]) + 1
    b = np.flatnonzero(sp[:-1] & ~sp[1:]) + 1
    for s, e in zip(a, b[b > a[0]] if len(a) else []):
        first_half[s:(s + e) // 2] = True
    parts["set_piece"] = np.where(first_half, C.DEAD_W_SET_PIECE, 0.0)
    # kick-off shape: every player of each team in its own half (one taker each may stand on
    # the line), ball near the centre spot
    xy = z["player_xy"].astype(float)
    team = z["player_team"]
    length = float(z["pitch"][0])
    half_idx = z["half"].astype(int)
    sign = z["attack_sign"][half_idx]                 # (n, 2): +1 attacks +x
    x_att = np.where(team == 0, np.where(sign[:, :1] > 0, xy[..., 0], length - xy[..., 0]),
                     np.where(team == 1, np.where(sign[:, 1:] > 0, xy[..., 0],
                                                  length - xy[..., 0]), np.nan))
    over = np.nansum(x_att > length / 2 + 1.0, axis=1)   # players beyond the halfway line
    n_on = np.sum(team >= 0, axis=1)
    bc = np.hypot(*(z["ball_xy"].astype(float) - np.array([length / 2,
                                                            float(z["pitch"][1]) / 2])).T)
    # ball at the centre spot, or not seen (the tracker often loses it at a kick-off)
    ball_ok = np.isnan(bc) | (bc < 3.0)
    shape = (over <= 2) & (n_on >= 18) & ball_ok
    parts["kickoff_shape"] = np.where(shape, C.DEAD_W_KICKOFF_SHAPE, 0.0)
    llr = sum(parts.values())
    return llr, parts


def before_kickoff(z, F):
    """Frames of each half before its kick-off restart (Smart Tagging 'Kick Off (Start)')."""
    sp = F["st_set_piece"]
    half = z["half"]
    from ..common import SET_PIECE_TYPES
    ko = SET_PIECE_TYPES.index("Kick Off (Start)") + 1
    mask = np.zeros(len(half), bool)
    speed = np.nan_to_num(F["ball_speed"].astype(float), nan=-1)
    for h in (1, 2):
        idx = np.flatnonzero((half == h) & (sp == ko))
        if not len(idx):
            continue
        a, b = idx[0], idx[-1] + 1
        # the kick: first frame in the window where the ball accelerates from rest
        kick = None
        for f in range(a, b - 1):
            if 0 <= speed[f] < 1.5 and speed[f + 1] >= 1.5:
                kick = f
                break
        if kick is None:
            kick = (a + b) // 2
        start = np.flatnonzero(half == h)[0]
        mask[start:kick] = True
    return mask


def min_durations(state, half, min_dead, min_alive):
    """Flip stoppages shorter than min_dead to play and gaps of play shorter than min_alive to
    stoppage, shortest first."""
    state = state.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        lo, hi = idx[0], idx[-1] + 1
        changed = True
        while changed:
            changed = False
            seg = state[lo:hi]
            b = np.r_[0, np.flatnonzero(np.diff(seg) != 0) + 1, len(seg)]
            spans = [(b[i + 1] - b[i], b[i], b[i + 1], seg[b[i]]) for i in range(len(b) - 1)]
            # interior spans only: the first and last of a half are bounded by the half itself
            interior = [s for s in spans[1:-1]
                        if s[0] < (min_dead if s[3] == 1 else min_alive)]
            if interior:
                L, a_, e_, v = min(interior)
                state[lo + a_:lo + e_] = 1 - v
                changed = True
    return state


def viterbi2(llr, half, cost):
    """Most likely alive(0)/dead(1) sequence for per-frame log-odds llr, with a switch cost.
    Each half is decoded on its own; frames outside the halves are left dead."""
    out = np.ones(len(llr), np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        e = llr[idx]
        n = len(e)
        # scores: alive gets -e/2, dead gets +e/2 (symmetric log-odds)
        s_alive = np.empty(n)
        s_dead = np.empty(n)
        back = np.zeros((n, 2), np.int8)
        s_alive[0], s_dead[0] = -e[0] / 2, e[0] / 2
        for i in range(1, n):
            a_stay, a_switch = s_alive[i - 1], s_dead[i - 1] - cost
            d_stay, d_switch = s_dead[i - 1], s_alive[i - 1] - cost
            if a_stay >= a_switch:
                s_alive[i], back[i, 0] = a_stay - e[i] / 2, 0
            else:
                s_alive[i], back[i, 0] = a_switch - e[i] / 2, 1
            if d_stay >= d_switch:
                s_dead[i], back[i, 1] = d_stay + e[i] / 2, 1
            else:
                s_dead[i], back[i, 1] = d_switch + e[i] / 2, 0
        state = np.zeros(n, np.int8)
        state[-1] = 1 if s_dead[-1] > s_alive[-1] else 0
        for i in range(n - 1, 0, -1):
            state[i - 1] = back[i, state[i]]
        out[idx] = state
    return out


def restart_moments(z, F, kickoff_shape=None):
    """Frames where play visibly restarts: the ball accelerates from rest, re-appears already
    moving, or (when the tracker cannot see the ball) a kick-off formation breaks up."""
    speed = F["ball_speed"].astype(float)
    s = np.nan_to_num(speed, nan=-1)
    n = len(s)
    rest = (s >= 0) & (s < 1.5)
    cand = []
    for f in np.flatnonzero(rest[:-1] & (s[1:] >= 1.5)):
        if f + 12 < n and np.nanmax(speed[f:f + 12]) > 4:
            cand.append(f)
    gl = z["ball_gap_len"]
    found = np.flatnonzero((gl[:-1] > 0) & (gl[1:] == 0)) + 1
    for f in found:
        if np.nanmean(speed[f:f + 8]) > 3:
            cand.append(int(f))
    if kickoff_shape is not None:
        ks = kickoff_shape > 0
        # the shape holds for a while, then breaks: that is the kick
        ends = np.flatnonzero(ks[:-1] & ~ks[1:]) + 1
        for f in ends:
            if ks[max(0, f - 50):f].mean() > 0.5:
                cand.append(int(f))
    return np.array(sorted(set(cand)), int)


def stop_moments(z, F):
    """Frames where play visibly stops: the ball crosses a line, or the tracker loses it."""
    out = np.nan_to_num(F["ball_outside"].astype(float))
    cross = np.flatnonzero((out[:-1] == 0) & (out[1:] > 0)) + 1
    gl = z["ball_gap_len"]
    lost = np.flatnonzero((gl[:-1] == 0) & (gl[1:] > 0)) + 1
    return np.array(sorted(set(cross) | set(lost)), int)


def snap(state, half, restarts, stops, window):
    """Move each boundary to the nearest restart (dead->alive) or stop (alive->dead) moment."""
    state = state.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        lo, hi = idx[0], idx[-1] + 1
        seg = state[lo:hi]
        bounds = np.flatnonzero(np.diff(seg) != 0) + 1 + lo
        for b in bounds:
            to_alive = state[b] == 0
            pool = restarts if to_alive else stops
            near = pool[(pool >= b - window) & (pool <= b + window)]
            if not len(near):
                continue
            t = int(near[np.argmin(np.abs(near - b))])
            if t > b:      # extend the previous state forward
                state[b:t] = state[b - 1]
            elif t < b:    # pull the new state back
                state[t:b] = state[b]
    return state


def enter_pitch(state, half, F, max_wait=50):
    """Play cannot restart while the ball is beyond a line (a throw-in starts outside; play
    resumes when the ball comes in). Push each restart to the first frame the ball is inside."""
    out = np.nan_to_num(F["ball_outside"].astype(float))
    state = state.copy()
    b = np.flatnonzero((state[:-1] == 1) & (state[1:] == 0)) + 1
    for f in b:
        if half[f] == 0 or out[f] <= C.DEAD_OUTSIDE_M:
            continue
        k = f
        while k < len(out) - 1 and k - f < max_wait and out[k] > C.DEAD_OUTSIDE_M:
            k += 1
        state[f:k] = 1
    return state


def kickoff_breaks(state, half, kickoff_shape, max_ahead=500):
    """A kick-off formation that holds for 2 s and then breaks up is the kick-off itself: play
    is on from there even if the ball is not seen yet."""
    ks = kickoff_shape > 0
    state = state.copy()
    for f in np.flatnonzero(ks[:-1] & ~ks[1:]) + 1:
        if half[f] == 0 or ks[max(0, f - 50):f].mean() <= 0.5 or state[f] == 0:
            continue
        nxt = np.flatnonzero(state[f:f + max_ahead] == 0)
        if len(nxt):
            state[f:f + nxt[0]] = 0
    return state


def detect(z, F, L2):
    half = z["half"]
    llr, parts = evidence(z, F, L2)
    state = viterbi2(llr, half, C.DEAD_SWITCH_COST)
    state = snap(state, half, restart_moments(z, F, parts["kickoff_shape"]), stop_moments(z, F),
                 C.DEAD_SNAP_WINDOW)
    state = kickoff_breaks(state, half, parts["kickoff_shape"])
    state = enter_pitch(state, half, F)
    state = min_durations(state, half, int(C.DEAD_MIN_DEAD_S * 25), int(C.DEAD_MIN_ALIVE_S * 25))
    state[before_kickoff(z, F)] = 1
    state[half == 0] = 1
    return state, llr, parts
