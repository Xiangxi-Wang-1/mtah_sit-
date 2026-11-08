"""Layer 4 - self-trained possession model.

    python -m possession.layer4_possession_model.run     # writes cache/layer4_tactical.npz

Input : cache/layer0..layer3
Output: p_home_players (players only - the evidence through ball gaps; an even blend of a
        moment model and a context model), p_home_full_self (players + ball): out-of-fold
        P(home has the ball) per graded frame from the self-trained model, label (the
        certain-possession frames it learned from: 1 home, 0 away, -1 none).
        p_home_full: P(home has the ball) from the cross-game model (crossgame.py), trained on
        every live frame of the bench games - never on this game.
        cp_frames, cp_p: candidate frames and P(no change / change to home / change to away)
        there, from the cross-game change-point model.
Needs the bench (python -m possession.bench.build, then layers 1-3 on each bench game);
python -m possession.bench.score does that in the right order. Trained models are cached in
cache/models/ (derived from other games; gitignored).
Gitignored derived data.
"""

import os
import time
import warnings

import numpy as np

from ..common import CACHE, ROOT, TAG, cache_path, load_layer
from . import crossgame as X
from .model import PARAMS, design, labels, out_of_fold

MODELS = os.path.join(CACHE, "models")


def bench_tags():
    d = os.path.join(CACHE, "bench")
    return sorted(f[:-7] for f in os.listdir(d) if f.endswith("_gt.csv")) if os.path.isdir(d) else []


def truth(tag):
    with open(os.path.join(CACHE, "bench", tag + "_gt.csv")) as f:
        f.readline()
        g = np.array([line.rstrip("\n").split(",")[1] for line in f])
    return np.where(g == "H", 0, np.where(g == "A", 1, np.where(g == "D", 2, -1))).astype(np.int8)


def full_matrix(tag):
    """Cross-game feature matrix of one game, cached (derived data; gitignored)."""
    path = os.path.join(MODELS, "design_{}.npz".format(tag))
    if os.path.isfile(path):
        d = np.load(path)
        return d["X"], list(d["names"])
    z, F, L2 = (load_layer(x, tag) for x in ("layer0", "layer1", "layer2"))
    S, base = X.sources(z, F, dyn=False)
    M, names = X.full_design(S, base, F, L2)
    os.makedirs(MODELS, exist_ok=True)
    np.savez(path, X=M, names=np.array(names))
    return M, names


def cross_game(verbose=True):
    """p_home_full, cp_frames, cp_p for the current game from models trained on the others."""
    t0 = time.time()
    train = [t for t in bench_tags() if t != TAG]
    if not train:
        raise IOError("no bench games: run python -m possession.bench.build (see bench/README.md)")
    key = "+".join(train)
    mats = {t: full_matrix(t) for t in train + [TAG]}
    names = mats[TAG][1]
    ys = {t: truth(t) for t in train}
    frame = X.cached(os.path.join(MODELS, "frame_{}.pkl".format(key)), lambda: X.fit(
        [mats[t][0] for t in train], [np.where(ys[t] == 0, 1, np.where(ys[t] == 1, 0, -1)) for t in train],
        names))
    p_full = X.predict(frame[0], frame[1], mats[TAG][0]).astype(np.float32)

    def cp_data(t):
        half = load_layer("layer0", t)["half"]
        fr = X.cp_candidates(load_layer("layer3", t)["dead"], half)
        Xc, cn = X.cp_design(mats[t][0], names, load_layer("layer2", t), fr)
        return fr, Xc, cn
    cp_train = {t: cp_data(t) for t in train}
    fr, Xc, cn = cp_data(TAG)
    cp = X.cached(os.path.join(MODELS, "cp_{}.pkl".format(key)), lambda: X.fit_cp(
        [cp_train[t][1] for t in train], [X.cp_labels(ys[t], cp_train[t][0]) for t in train], cn))
    cp_p = X.predict_cp(cp[0], cp[1], Xc).astype(np.float32)
    if verbose:
        print("cross-game models trained on {}  ({:.0f}s)".format(key, time.time() - t0))
    return {"p_home_full": p_full, "cp_frames": fr.astype(np.int32), "cp_p": cp_p}

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True, with_passes=True, params=None):
    t0 = time.time()
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    L3 = load_layer("layer3")
    y, seg = labels(z, F, L2, L3, with_passes=with_passes)
    params = params or PARAMS
    out = {"label": y, "label_segment": seg}
    p = {}
    for view in ("players", "players_context", "full"):
        X, _ = design(F, view)
        p[view] = out_of_fold(X, y, z["half"], params)
    out["p_home_players"] = 0.5 * p["players"] + 0.5 * p["players_context"]
    out["p_home_full_self"] = p["full"]
    out.update(cross_game(verbose))
    if verbose:
        print("labels: {} frames ({:.0%} home)  ({:.0f}s)".format(
            int((y >= 0).sum()), float(np.mean(y[y >= 0] == 1)), time.time() - t0))
    return out


def main():
    out = build()
    path = cache_path("layer4")
    np.savez_compressed(path, **out)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
