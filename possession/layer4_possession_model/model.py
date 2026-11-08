"""Self-trained possession model: P(home has the ball) per frame.

Labels come from the match itself, where possession is physically certain: a player controlling
the ball for >= 1 s with no opponent within reach (layer 2), in play (layer 3) - and optionally the
ball travelling between two such controls of the same team (a completed pass). An algorithm
picks them; no person labels anything, which the rules allow.

Two views, because they answer different questions:
  players  players-only features (shape, pressing, movement) at several time scales. Needs no
           ball, so it is the evidence through the 41% of frames where the ball is missing.
  full     players + ball features. The strongest evidence where the ball is seen.

Every prediction is out-of-fold: the match is cut into contiguous time blocks and each block is
predicted by a model that never saw it (nor POSS_BUFFER frames either side), so the output is
honest everywhere, including on the labelled frames.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import config as C
from ..layer1_features.features import rolling_mean
from ..layer2_touches.touches import RECEIVE

PLAYER_BASE = ["territory", "spread_x_h", "spread_x_a", "spread_y_h", "spread_y_a", "area_h",
               "area_a", "centroid_fwd_speed_h", "centroid_fwd_speed_a", "press_h", "press_a",
               "n_opp_half_h", "n_opp_half_a", "fwd_share_h", "fwd_share_a", "length_h",
               "length_a", "def_line_h", "def_line_a", "mean_speed_h", "mean_speed_a"]
BALL_BASE = ["margin", "d_h", "d_a", "relv_h", "relv_a", "closing_h", "closing_a", "n2_h", "n2_a",
             "n5_h", "n5_a", "nearest_fwd_speed_h", "nearest_fwd_speed_a", "ball_x_home_frame",
             "ball_ahead_h", "ball_ahead_a", "ball_fwd_speed_home", "ball_fwd_1s", "margin_1s",
             "margin_3s", "margin_10s", "near_share_1s", "near_share_3s", "near_share_10s"]
SCALES = ((37, "3s"), (125, "10s"))
# The players view is an even blend of two models (chosen in evaluate.py --variants): one sees
# the moment (plus 3 s / 10 s means), the other also sees 30 s means and the 3 s mean 5 s before
# and after. Context alone wins on held-out certain moments but loses on stale frames; the blend
# beats both on every measure.
CONTEXT_SCALES = ((37, "3s"), (125, "10s"), (750, "30s"))
CONTEXT_LAGS = (-125, 125)
PARAMS = {"leaves": 31}


def labels(z, F, L2, L3, with_passes=True):
    """(label per frame: 1 home, 0 away, -1 none; segment id per labelled frame)."""
    n = len(z["half"])
    y = np.full(n, -1, np.int8)
    seg = np.full(n, -1, np.int32)
    E = L2["events"]
    rec = np.sort(E[E["kind"] == RECEIVE], order="frame")
    ok = rec[(rec["duration"] >= C.POSS_LABEL_MIN_CONTROL) & (rec["contest"] <= C.POSS_LABEL_MAX_CONTEST)]
    alive = (L3["dead"] == 0) & (z["half"] > 0)
    for k, r in enumerate(ok):
        a, b = int(r["frame"]), int(r["frame"] + r["duration"])
        m = np.arange(a, b)[alive[a:b]]
        y[m] = 1 if r["team"] == 0 else 0
        seg[m] = k
    if with_passes:
        allrec = rec[rec["contest"] <= C.POSS_LABEL_MAX_CONTEST]
        for p, q in zip(allrec[:-1], allrec[1:]):
            a = int(p["frame"] + p["duration"])
            b = int(q["frame"])
            if p["team"] != q["team"] or b - a > C.POSS_LABEL_PASS_GAP or b <= a:
                continue
            if not alive[a:b].all():
                continue
            y[a:b] = 1 if p["team"] == 0 else 0
            seg[a:b] = np.where(seg[a:b] >= 0, seg[a:b], 100000 + int(a))
    return y, seg


def design(F, view):
    """Feature matrix for a view: "players", "players_context" or "full"."""
    cols, names = [], []
    base = PLAYER_BASE + BALL_BASE if view == "full" else PLAYER_BASE
    scales = CONTEXT_SCALES if view == "players_context" else SCALES
    lags = CONTEXT_LAGS if view == "players_context" else ()
    for f in base:
        x = F[f].astype(float)
        cols.append(x)
        names.append(f)
        if f in PLAYER_BASE:
            for w, tag in scales:
                cols.append(rolling_mean(x, w))
                names.append("{}_{}".format(f, tag))
            for L in lags:
                m = rolling_mean(x, 37)
                sh = np.full_like(m, np.nan)
                if L > 0:
                    sh[:-L] = m[L:]
                else:
                    sh[-L:] = m[:L]
                cols.append(sh)
                names.append("{}_3s_{:+d}".format(f, L))
    return np.column_stack(cols).astype(np.float32), names


def blocks(half, k):
    """Contiguous blocks of graded frames: list of (start, end) video frames."""
    idx = np.flatnonzero(half > 0)
    parts = np.array_split(idx, k)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


def out_of_fold(X, y, half, params=None, k=None, buffer=None):
    """Out-of-fold P(home) for every frame. Returns (p, models)."""
    k = k or C.POSS_FOLDS
    buffer = C.POSS_BUFFER if buffer is None else buffer
    params = params or {}
    n = len(y)
    p = np.full(n, np.nan, np.float32)
    lab = y >= 0
    for a, b in blocks(half, k):
        train = lab.copy()
        train[max(0, a - buffer):min(n, b + buffer)] = False
        clf = HistGradientBoostingClassifier(max_iter=params.get("max_iter", 300),
                                             learning_rate=params.get("lr", 0.05),
                                             max_leaf_nodes=params.get("leaves", 15),
                                             min_samples_leaf=params.get("min_leaf", 200),
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X[train], y[train])
        p[a:b] = clf.predict_proba(X[a:b])[:, 1]
    return p
