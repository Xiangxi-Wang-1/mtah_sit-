"""Evaluate layer 2 without any ground truth.

    python -m possession.layer2_touches.evaluate          # reads cache/layer2_tactical.npz
    python -m possession.layer2_touches.evaluate --sweep  # also sweeps the main thresholds

Three self-supervised questions:

  precise?    a touch should happen next to a player. Free touches are found from the ball's
              motion alone, so the distance to the nearest player is an independent check.
  complete?   between two consecutive events the ball is free, and a free ball goes straight
              and only slows down. A stretch where it speeds up or turns without an event in
              between is a missed touch.
  plausible?  football constants: touches per minute, how often the next player to control the
              ball is a team-mate (pass completion is ~75-90% in top leagues), pass lengths,
              how long a player keeps the ball.

Writes reports/layer2/report.md.
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np

from .. import config as C
from ..common import REPORTS, ROOT, load_layer
from .touches import (ARTIFACT, FOUND, KIND_NAMES, LOST, RECEIVE, RELEASE, STOP, TOUCH,
                      UNATTRIBUTED, detect, heading_change)

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer2")


def completeness(E, frames, z):
    """Fraction of free-ball stretches that speed up or turn with no event to explain it."""
    half = z["half"]
    speed = z["ball_speed"].astype(float)
    vel = z["ball_vel"].astype(float)
    state = frames["ball_state"]
    ev = np.sort(E["frame"])
    free = (state == 2) & (half > 0)
    m = np.concatenate(([False], free, [False])).astype(np.int8)
    d = np.diff(m)
    A, B = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    n = sped = turned = 0
    for a, b in zip(A, B):
        # split the free stretch at events inside it
        inner = ev[(ev > a) & (ev < b)]
        for s, e in zip(np.r_[a, inner], np.r_[inner, b]):
            s, e = s + 3, e - 3            # ignore the edges, where the touch itself is
            if e - s < 10:
                continue
            n += 1
            sp = speed[s:e]
            # the kick that started the stretch accelerates the (smoothed) ball for a while:
            # only a new rise after the speed has started falling is unexplained
            peak = int(np.argmax(sp[:min(len(sp), 20)]))
            sp = sp[peak:]
            rise = np.max(sp - np.minimum.accumulate(sp)) if len(sp) else 0
            if rise > 2.5:
                sped += 1
            v0 = np.nanmean(vel[s:s + 4], axis=0)
            v1 = np.nanmean(vel[e - 4:e], axis=0)
            if np.hypot(*v0) > 2 and np.hypot(*v1) > 2:
                cos = np.dot(v0, v1) / (np.hypot(*v0) * np.hypot(*v1))
                if np.degrees(np.arccos(np.clip(cos, -1, 1))) > 30:
                    turned += 1
    return {"free_stretches": n, "unexplained_speedup": sped / max(n, 1),
            "unexplained_turn": turned / max(n, 1),
            "unexplained_any": (sped + turned) / max(n, 1)}


def precision(E, F, z, state):
    """How close the nearest player is at free touches, against random ball frames."""
    k = E["kind"]
    free = E[(k == TOUCH) | (k == UNATTRIBUTED)]
    d = np.fmin(F["d_h"], F["d_a"]).astype(float)
    from scipy.ndimage import minimum_filter1d
    dwin = minimum_filter1d(np.nan_to_num(d, nan=99), 2 * C.TOUCH_ATTRIB_WINDOW + 1)
    # baseline: random frames where the ball is free (not with a player), like free touches
    g = (z["half"] > 0) & (state == 2)
    rnd = np.random.default_rng(0).choice(np.flatnonzero(g), 5000)
    out = {"n_free": int(len(free)),
           "attributed_share": float(np.mean(free["kind"] == TOUCH)) if len(free) else None}
    for r in (1.0, 1.5, 2.5):
        out["near_{}".format(r).replace(".", "_")] = float(np.mean(dwin[free["frame"]] < r)) if len(free) else None
        out["random_near_{}".format(r).replace(".", "_")] = float(np.mean(dwin[rnd] < r))
    return out


def plausibility(E, frames, z):
    half = z["half"]
    known_min = float(((half > 0) & ~np.isnan(z["ball_xy"][:, 0])).sum() / 1500.0)
    k = E["kind"]
    rec = E[k == RECEIVE]
    rel = E[k == RELEASE]
    touch = E[k == TOUCH]
    # chain of attributed ball-players, in time order
    chain = np.sort(E[(k == RECEIVE) | (k == TOUCH)], order="frame")
    same = [a["team"] == b["team"] for a, b in zip(chain[:-1], chain[1:])
            if b["frame"] - a["frame"] < 10 * 25 and half[a["frame"]] == half[b["frame"]]]
    rec_same = [a["team"] == b["team"] for a, b in zip(rec[:-1], rec[1:])
                if b["frame"] - a["frame"] < 10 * 25 and half[a["frame"]] == half[b["frame"]]]
    # pass length: release -> next receive
    bxy = z["ball_xy"].astype(float)
    lengths = []
    for r in rel:
        nxt = rec[rec["frame"] > r["frame"]]
        if len(nxt) and nxt[0]["frame"] - r["frame"] < 5 * 25:
            lengths.append(np.hypot(*(bxy[nxt[0]["frame"]] - bxy[r["frame"]])))
    held = (rel["frame"] - rec["frame"]) / 25.0 if len(rel) == len(rec) else np.array([])
    ctrl_share = float(np.mean(frames["ball_state"][(half > 0) & (frames["ball_state"] > 0)] == 1))
    return {
        "ball_known_minutes": known_min,
        "receives_per_min": len(rec) / known_min, "free_touches_per_min": len(touch) / known_min,
        "next_toucher_same_team": float(np.mean(same)) if same else None,
        "next_receiver_same_team": float(np.mean(rec_same)) if rec_same else None,
        "pass_length_m_p25_p50_p75": [float(np.percentile(lengths, q)) for q in (25, 50, 75)]
        if lengths else None,
        "held_s_p25_p50_p75": [float(np.percentile(held, q)) for q in (25, 50, 75)]
        if len(held) else None,
        "share_of_known_ball_with_a_player": ctrl_share,
    }


def evaluate_once(z, F, E=None, frames=None):
    if E is None:
        E, frames = detect(z, F)
    return {"counts": {KIND_NAMES[i]: int((E["kind"] == i).sum()) for i in range(len(KIND_NAMES))},
            "precision": precision(E, F, z, frames["ball_state"]), "completeness": completeness(E, frames, z),
            "plausibility": plausibility(E, frames, z)}


def sweep(z, F):
    """Main thresholds, one at a time around the defaults (each run takes ~1 s)."""
    grid = {
        "CONTROL_D": [1.0, 1.25, 1.5, 2.0],
        "CONTROL_RELV": [2.0, 3.0, 4.0],
        "TOUCH_SPEED_MIN_PROMINENCE": [1.0, 2.0, 3.0],
        "TOUCH_TURN_DEG": [20.0, 30.0, 45.0],
        "TOUCH_TURN_MAX_DEG": [60.0, 90.0, 120.0],
    }
    rows = []
    for name, values in grid.items():
        old = getattr(C, name)
        for v in values:
            setattr(C, name, v)
            r = evaluate_once(z, F)
            rows.append({"param": name, "value": v,
                         "unexplained_any": r["completeness"]["unexplained_any"],
                         "near_1_5": r["precision"]["near_1_5"], "random_1_5": r["precision"]["random_near_1_5"],
                         "attributed": r["precision"]["attributed_share"],
                         "next_receiver_same_team": r["plausibility"]["next_receiver_same_team"],
                         "receives_per_min": r["plausibility"]["receives_per_min"],
                         "free_touches_per_min": r["plausibility"]["free_touches_per_min"]})
        setattr(C, name, old)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    z = load_layer("layer0")
    F = {k: (v.astype(float) if v.dtype.kind == "f" else v) for k, v in load_layer("layer1").items()}
    L2 = load_layer("layer2")
    R = evaluate_once(z, F, L2["events"], {"ball_state": L2["ball_state"]})
    if args.sweep:
        R["sweep"] = sweep(z, F)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "report.md"), "w") as f:
        f.write("# Layer 2 evaluation\n\n```\n" + json.dumps(R, indent=1, default=float) + "\n```\n")
    print(json.dumps({k: v for k, v in R.items() if k != "sweep"}, indent=1, default=float))
    if args.sweep:
        for r in R["sweep"]:
            print("{param:28} {value:5}  unexplained {unexplained_any:.3f}  near1.5 {near_1_5:.3f} (rand {random_1_5:.3f})  "
                  "attrib {attributed:.2f}  same-team {next_receiver_same_team:.3f}  "
                  "recv/min {receives_per_min:.1f}  touch/min {free_touches_per_min:.1f}".format(**r))


if __name__ == "__main__":
    main()
