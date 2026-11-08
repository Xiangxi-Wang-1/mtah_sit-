"""Possession without the ball, trained on other games with true labels (the bench).

The self-trained model (model.py) learns only from moments where possession is physically
certain on our own match: calm controls, where the answer is easy anyway. The bench games
(possession/bench) have an answer key on *every* live frame, including the first seconds after a
turnover and long ball gaps, which are exactly where layers 2-5 go wrong. So this model is
trained on every live frame of the other games:

  bench game g    trained on the two other bench games, never on g (leave one game out)
  challenge match trained on all three bench games; nothing from the challenge match itself

Features are players-only (no ball): the layer-1 shape / pressing / movement features, extra
shape features (line quantiles, width, sprinting / retreating shares, front line vs the
opponent's defensive line) and, optionally, how those change over the next seconds (a team that
has just won the ball pushes forward; one that lost it drops back). Every row is also added with
home and away swapped and the label flipped, so the model learns "the team with the ball", not
"this particular home team", and its prediction is averaged over both orientations.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..layer1_features.features import rolling_mean
from .model import PLAYER_BASE

EXTRA_BASE = ["xq10_h", "xq30_h", "xq50_h", "xq70_h", "xq90_h", "ywidth_h", "sprint_h", "vy_h",
              "back_share_h", "xq10_a", "xq30_a", "xq50_a", "xq70_a", "xq90_a", "ywidth_a",
              "sprint_a", "vy_a", "back_share_a", "front_vs_oppdef_h", "front_vs_oppdef_a"]
DYN_KEYS = ["centroid_fwd_speed_h", "centroid_fwd_speed_a", "fwd_share_h", "fwd_share_a",
            "back_share_h", "back_share_a", "sprint_h", "sprint_a", "press_h", "press_a",
            "n_opp_half_h", "n_opp_half_a", "territory", "spread_y_h", "spread_y_a", "area_h",
            "area_a"]
MOMENT = dict(scales=(37, 125), lags=())
CONTEXT = dict(scales=(37, 125, 750), lags=(-125, 125))
PARAMS = dict(max_iter=300, lr=0.05, leaves=31, min_leaf=200)


def extra_features(z, F):
    xy = z["player_xy"].astype(float)
    vel = z["player_vel"].astype(float)
    team = z["player_team"]
    half = z["half"]
    L = float(z["pitch"][0])
    sign = z["attack_sign"][half.astype(int)]
    X = {}
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t, tag in ((0, "h"), (1, "a")):
                mine = team == t
                s = sign[:, t][:, None]
                xa = np.where(mine, np.where(s > 0, xy[..., 0], L - xy[..., 0]), np.nan)
                ya = np.where(mine, xy[..., 1], np.nan)
                out = np.sort(xa, axis=1)[:, 1:]          # drop the deepest player (keeper)
                for q in (10, 30, 50, 70, 90):
                    X["xq{}_{}".format(q, tag)] = np.nanpercentile(out, q, axis=1)
                X["ywidth_" + tag] = np.nanpercentile(ya, 90, axis=1) - np.nanpercentile(ya, 10, axis=1)
                sp = np.hypot(vel[..., 0], vel[..., 1])
                nn = np.sum(mine & ~np.isnan(sp), axis=1)
                X["sprint_" + tag] = np.where(nn > 0, np.sum(mine & (sp > 4), axis=1) / np.maximum(nn, 1), np.nan)
                X["vy_" + tag] = np.nanmean(np.where(mine, np.abs(vel[..., 1]), np.nan), axis=1)
                fwd = vel[..., 0] * s
                X["back_share_" + tag] = np.where(nn > 0, np.sum(mine & (fwd < -1), axis=1) / np.maximum(nn, 1), np.nan)
    X["front_vs_oppdef_h"] = (L - F["def_line_a"].astype(float)) - F["front_line_h"].astype(float)
    X["front_vs_oppdef_a"] = (L - F["def_line_h"].astype(float)) - F["front_line_a"].astype(float)
    return X


def window_mean(x, lo, hi):
    """NaN-aware mean of x over [t+lo, t+hi)."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    cn = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    a, b = np.clip(i + lo, 0, len(x)), np.clip(i + hi, 0, len(x))
    c = cn[b] - cn[a]
    with np.errstate(all="ignore"):
        return np.where(c > 0, (cs[b] - cs[a]) / c, np.nan)


def sources(z, F, dyn=True):
    """name -> per-frame array for every base feature this model may use."""
    S = {k: F[k] for k in PLAYER_BASE}
    S.update(extra_features(z, F))
    base = list(PLAYER_BASE) + EXTRA_BASE
    if dyn:
        for k in DYN_KEYS:
            x = S[k].astype(float)
            past = window_mean(x, -75, 0)
            stem, team = (k[:-2], k[-2:]) if k.endswith(("_h", "_a")) else (k, "")
            for lo, hi, tag in ((0, 25, "f1"), (0, 75, "f3"), (25, 125, "f1_5")):
                S["{}_{}{}".format(stem, tag, team)] = window_mean(x, lo, hi)
            S["{}_d3{}".format(stem, team)] = window_mean(x, 0, 75) - past
            base += ["{}_{}{}".format(stem, t, team) for t in ("f1", "f3", "f1_5", "d3")]
    return S, base


def design(S, base, scales, lags):
    cols, names = [], []
    for f in base:
        x = S[f].astype(float)
        cols.append(x)
        names.append(f)
        for w in scales:
            cols.append(rolling_mean(x, w))
            names.append("{}@{}".format(f, w))
        for L in lags:
            m = rolling_mean(x, 37)
            sh = np.full_like(m, np.nan)
            if L > 0:
                sh[:-L] = m[L:]
            else:
                sh[-L:] = m[:L]
            cols.append(sh)
            names.append("{}@lag{}".format(f, L))
    return np.column_stack(cols).astype(np.float32), names


# signed home-frame quantities: swapping home and away negates them
FLIP = ("territory", "margin", "ball_fwd_speed_home")


def swap_index(names):
    """Column permutation and sign that swap home and away."""
    idx, sgn = [], []
    for nm in names:
        base = nm.split("@")[0]
        suf = nm[len(base):]
        if base.endswith(("_h", "_a")):
            other = base[:-2] + ("_a" if base.endswith("_h") else "_h") + suf
            idx.append(names.index(other))
            sgn.append(1.0)
        else:
            idx.append(names.index(nm))
            sgn.append(-1.0 if base.startswith(FLIP) else 1.0)
    return np.array(idx), np.array(sgn, np.float32)


def fit(Xs, ys, names, params=PARAMS, stride=3):
    """One classifier on several games' rows (every `stride`-th labelled frame), with the
    home/away-swapped copy of every row."""
    idx, sgn = swap_index(names)
    X = np.vstack([x[y >= 0][::stride] for x, y in zip(Xs, ys)])
    y = np.concatenate([y[y >= 0][::stride] for y in ys])
    X = np.vstack([X, X[:, idx] * sgn])
    y = np.r_[y, 1 - y]
    clf = HistGradientBoostingClassifier(max_iter=params["max_iter"], learning_rate=params["lr"],
                                         max_leaf_nodes=params["leaves"],
                                         min_samples_leaf=params["min_leaf"],
                                         l2_regularization=1.0, random_state=0)
    clf.fit(X, y)
    return clf, (idx, sgn)


def predict(clf, swap, X):
    idx, sgn = swap
    return 0.5 * (clf.predict_proba(X)[:, 1] + 1 - clf.predict_proba(X[:, idx] * sgn)[:, 1])


# --- the full view: players + ball, looking both ways in time ---------------------------------
BALL_PAIRS = ["d", "relv", "closing", "n2", "n5", "dwell", "nearest_fwd_speed", "ball_ahead"]
BALL_NEUTRAL = ["ball_known", "ball_speed", "ball_gap_len", "ball_frames_since_seen",
                "ball_frames_until_seen", "ball_to_line", "ball_outside"]
WINDOWS = ((-200, -75), (-75, -25), (-25, 0), (0, 25), (25, 75), (75, 200))


def _next_control(ctrl):
    """(team of the next control segment, frames until it) for every frame."""
    n = len(ctrl)
    nxt = np.full(n, -1, np.int8)
    until = np.full(n, 10000, np.int32)
    t, f = -1, -1
    for i in range(n - 1, -1, -1):
        if ctrl[i] >= 0:
            t, f = ctrl[i], i
        nxt[i] = t
        until[i] = f - i if f >= 0 else 10000
    return nxt, until


def ball_signals(F, L2):
    """name -> per-frame array; team-paired names end in _h / _a, signed ones are in FLIP."""
    S = {}
    for k in BALL_PAIRS:
        for t in ("h", "a"):
            S["{}_{}".format(k, t)] = F["{}_{}".format(k, t)].astype(float)
    S["margin"] = np.clip(F["margin"].astype(float), -10, 10)
    S["ball_fwd_speed_home"] = F["ball_fwd_speed_home"].astype(float)
    for k in BALL_NEUTRAL:
        S[k] = F[k].astype(float)
    ctrl = L2["control_team"].astype(int)
    last = L2["last_control_team"].astype(int)
    nxt, until = _next_control(ctrl)
    since = L2["frames_since_control"].astype(float)
    for t, tag in ((0, "h"), (1, "a")):
        S["ctrl_" + tag] = (ctrl == t).astype(float)
        S["lastctrl_" + tag] = (last == t).astype(float)
        S["nextctrl_" + tag] = (nxt == t).astype(float)
    S["since_ctrl"] = np.minimum(np.where(since < 0, 10000, since), 1000)
    S["until_ctrl"] = np.minimum(until, 1000).astype(float)
    return S


def full_design(Sp, base_p, F, L2):
    """Players (moment view) + ball signals at the frame and averaged over WINDOWS."""
    Xp, names = design(Sp, base_p, **MOMENT)
    S = ball_signals(F, L2)
    cols = []
    for k, x in S.items():
        stem, team = (k[:-2], k[-2:]) if k.endswith(("_h", "_a")) else (k, "")
        cols.append(x)
        names.append(k)
        for lo, hi in WINDOWS:
            cols.append(window_mean(x, lo, hi))
            names.append("{}_w{}_{}{}".format(stem, lo, hi, team).replace("-", "m"))
    return np.column_stack([Xp] + [np.column_stack(cols).astype(np.float32)]), names


# --- change points: where possession passes, and to whom ---------------------------------------
CP_WINDOW = 6          # frames: a candidate within this of a true change is labelled with it
CP_STEP = 3            # frames between candidates (every live frame on this grid)


def event_features(L2, n):
    """Layer-2 events (receptions, touches, kicks, ball found / lost, releases) of each team,
    counted in windows just before, at, and after each frame."""
    from ..layer2_touches.touches import FOUND, KICK, LOST, RECEIVE, RELEASE, TOUCH
    E = L2["events"]
    out = {}
    for kind, nm in ((RECEIVE, "rec"), (TOUCH, "touch"), (KICK, "kick"), (FOUND, "found"),
                     (LOST, "lost"), (RELEASE, "rel")):
        for t, tag in ((0, "_h"), (1, "_a")):
            m = (E["kind"] == kind) & (E["team"] == t)
            x = np.zeros(n)
            x[E["frame"][m]] = 1
            for lo, hi in ((-12, -3), (-3, 4), (4, 13), (13, 50)):
                out["ev_{}_{}_{}{}".format(nm, lo, hi, tag).replace("-", "m")] = window_mean(x, lo, hi)
    return out


def cp_candidates(dead, half):
    alive = (dead == 0) & (half > 0)
    return np.flatnonzero(alive & (np.arange(len(half)) % CP_STEP == 0))


def cp_design(Xfull, names, L2, frames):
    ef = event_features(L2, len(Xfull))
    keys = list(ef)
    X = np.column_stack([Xfull[frames]] + [np.column_stack([ef[k][frames] for k in keys])])
    return X.astype(np.float32), names + keys


def cp_labels(truth, frames):
    """0 no change, 1 change to home, 2 change to away, within CP_WINDOW of each candidate."""
    n = len(truth)
    ch = np.zeros(n, np.int8)
    ha = np.isin(truth, [0, 1])
    for i in np.flatnonzero(ha[1:] & ha[:-1] & (truth[1:] != truth[:-1])) + 1:
        ch[max(0, i - CP_WINDOW):i + CP_WINDOW + 1] = 1 if truth[i] == 0 else 2
    return ch[frames]


def fit_cp(Xs, ys, names, params=PARAMS):
    idx, sgn = swap_index(names)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    ysw = np.where(y == 1, 2, np.where(y == 2, 1, 0))
    clf = HistGradientBoostingClassifier(max_iter=params["max_iter"], learning_rate=params["lr"],
                                         max_leaf_nodes=params["leaves"], min_samples_leaf=100,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(np.vstack([X, X[:, idx] * sgn]), np.r_[y, ysw])
    return clf, (idx, sgn)


def predict_cp(clf, swap, X):
    idx, sgn = swap
    return 0.5 * (clf.predict_proba(X) + clf.predict_proba(X[:, idx] * sgn)[:, [0, 2, 1]])


# --- training on the bench, with a model cache ---------------------------------------------------
def cached(path, build):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj
