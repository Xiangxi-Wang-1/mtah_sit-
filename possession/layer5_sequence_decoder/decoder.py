"""Event-anchored semi-Markov decoder: one clean H / A / D timeline per half.

Dead ball comes fixed from layer 3. Each spell of live play between two stoppages is decoded on
its own:

  1. candidate change points: the start of every reception (layer 2), every kick (the ball
     accelerating away from a player who touched it briefly), every touch attributed to a player, every frame the ball re-appears after a gap, and one point a second inside ball
     gaps longer than 2 s (a hand-over can hide in a gap). Possession may change only there.
  2. evidence: a per-frame log-odds that home has the ball - from controls (layer 2) and the
     players-only possession model (layer 4) - summed over each stretch between two points.
  3. semi-Markov dynamic programming over the points: a possession is a run of stretches with
     one team; its score is its evidence, plus a log-normal prior on its length, minus a switch
     cost at its start that a reception or close touch by the new team pays back. The best
     sequence of possessions for the spell is found exactly (O(K^2) in the number of points).
"""

import numpy as np

from .. import config as C
from ..layer2_touches.touches import FOUND, KICK, RECEIVE, TOUCH


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def frame_evidence(L2, L4):
    """Per-frame log-odds that home has the ball."""
    n = len(L2["ball_state"])
    ctrl = np.zeros(n)
    E = L2["events"]
    for r in E[E["kind"] == RECEIVE]:
        a, b = int(r["frame"]), int(r["frame"] + r["duration"])
        w = C.SEQ_W_CONTROL * (1.0 - float(np.nan_to_num(r["contest"])))
        ctrl[a:b] = w if r["team"] == 0 else -w
    llr = ctrl.copy()
    llr += C.SEQ_W_PLAYERS * np.nan_to_num(logit(L4["p_home_players"].astype(float)))
    if C.SEQ_W_FULL:
        llr += C.SEQ_W_FULL * np.nan_to_num(logit(L4["p_home_full"].astype(float)))
    return llr


def candidate_points(L2, z, L4=None):
    """{frame: (bonus if the new team is home, bonus if away)} - where possession may change."""
    E = L2["events"]
    pts = {}

    def add(f, bh, ba):
        oh, oa = pts.get(f, (0.0, 0.0))
        pts[f] = (max(oh, bh), max(oa, ba))

    for r in E[E["kind"] == RECEIVE]:
        b = C.SEQ_W_RECEIVE * (1 - float(np.nan_to_num(r["contest"]))) * min(1.0, r["duration"] / 25.0)
        add(int(r["frame"]), b if r["team"] == 0 else 0.0, b if r["team"] == 1 else 0.0)
    for t in E[E["kind"] == TOUCH]:
        b = C.SEQ_W_TOUCH if t["dist"] <= 1.0 else 0.0
        add(int(t["frame"]), b if t["team"] == 0 else 0.0, b if t["team"] == 1 else 0.0)
    for k in E[E["kind"] == KICK]:
        b = C.SEQ_W_KICK
        add(int(k["frame"]), b if k["team"] == 0 else 0.0, b if k["team"] == 1 else 0.0)
    for f in E[E["kind"] == FOUND]["frame"]:
        add(int(f), 0.0, 0.0)
    if C.SEQ_W_CP and L4 is not None and "cp_frames" in L4:
        # learned change points: log-odds that possession passes to home / away here
        p = np.clip(L4["cp_p"].astype(float), 1e-4, 1)
        for f, (p0, ph, pa) in zip(L4["cp_frames"], p):
            oh, oa = pts.get(int(f), (0.0, 0.0))       # added to any event bonus here
            pts[int(f)] = (oh + C.SEQ_W_CP * np.log(ph / p0) + C.SEQ_CP_OFFSET,
                           oa + C.SEQ_W_CP * np.log(pa / p0) + C.SEQ_CP_OFFSET)
    if C.SEQ_DENSE_GRID:                 # anywhere on a grid: lets the frame evidence alone switch
        for f in range(0, len(z["half"]), C.SEQ_DENSE_GRID):
            add(f, 0.0, 0.0)
    gl = z["ball_gap_len"]
    gap = gl > 50
    a = np.flatnonzero(gap[1:] & ~gap[:-1]) + 1
    b = np.flatnonzero(gap[:-1] & ~gap[1:]) + 1
    for s, e in zip(a, b[b > a[0]] if len(a) else []):
        for f in range(s + C.SEQ_GAP_GRID, e, C.SEQ_GAP_GRID):
            add(int(f), 0.0, 0.0)
    return pts


def duration_prior(frames):
    L = np.maximum(frames, 1) / 25.0
    return -C.SEQ_W_DUR * (np.log(L) - np.log(C.SEQ_DUR_MEDIAN_S)) ** 2 / (2 * C.SEQ_DUR_SIGMA ** 2)


def decode_spell(a, b, cum, pts):
    """Best H(0)/A(1) labelling of frames [a, b) given cumulative evidence and points."""
    inner = sorted(f for f in pts if a < f < b)
    bounds = [a] + inner + [b]
    K = len(bounds) - 1
    B = np.array(bounds)
    S = cum[B]                                 # S[j] - S[i] = evidence of frames bounds[i]..bounds[j]
    NEG = -1e18
    bon = np.zeros((K + 1, 2))
    for k in range(1, K):
        bon[k] = pts[B[k]]
    f = np.full((K + 1, 2), NEG)
    arg = np.zeros((K + 1, 2), int)
    for j in range(1, K + 1):
        i = np.arange(j)
        ev = S[j] - S[i]
        dur = duration_prior(B[j] - B[i])
        for t in (0, 1):
            sgn = 0.5 if t == 0 else -0.5
            seg = sgn * ev + dur
            prev = np.where(i == 0, 0.0, f[i, 1 - t])
            cost = np.where(i == 0, 0.0, C.SEQ_SWITCH_COST - bon[i, t])
            tot = prev + seg - cost
            k = int(np.argmax(tot))
            f[j, t], arg[j, t] = tot[k], k
    lab = np.zeros(b - a, np.int8)
    t = int(np.argmax(f[K]))
    j = K
    while j > 0:
        i = arg[j, t]
        lab[B[i] - a:B[j] - a] = t
        j, t = i, 1 - t
    return lab


def decode(z, L2, L3, L4):
    """Per frame: 0 home, 1 away, 2 dead (and outside the halves)."""
    half = z["half"]
    dead = L3["dead"].astype(bool) | (half == 0)
    llr = frame_evidence(L2, L4)
    cum = np.concatenate(([0.0], np.cumsum(llr)))
    pts = candidate_points(L2, z, L4)
    out = np.full(len(half), 2, np.int8)
    alive = ~dead
    a_ = np.flatnonzero(alive[1:] & ~alive[:-1]) + 1
    b_ = np.flatnonzero(alive[:-1] & ~alive[1:]) + 1
    if alive[0]:
        a_ = np.r_[0, a_]
    if alive[-1]:
        b_ = np.r_[b_, len(alive)]
    for a, b in zip(a_, b_):
        out[a:b] = decode_spell(int(a), int(b), cum, pts)
    return out
