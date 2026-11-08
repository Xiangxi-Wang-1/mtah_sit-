"""How well do layers 0-2 work together? A minimal possession read-out from the touches.

    python -m possession.layer2_touches.evaluate_chain

Not the final decoder (that is layer 5), just enough to see whether the touch layer carries the
right information. Rule: possession changes only when a player of the other team controls the
ball (a control segment of at least MIN_CONTROL frames); everything else - deflections, loose
balls, gaps - keeps the last team. Compared against the official baseline (nearest player,
+/- 50 frame majority vote) on three things that need no ground truth:

  - agreement with the Smart Tagging possession proxy (frames where a phase implies a side)
  - timing at the Smart Tagging turnovers (start of each ATTACKING TRANSITION)
  - possessions per minute and their length (football: roughly 2-4 per minute of play)
"""

import json
import os
import warnings

import numpy as np

from ..common import REPORTS, ROOT, load_layer, load_smart_tagging, team_index_from_names
from .touches import RECEIVE, TOUCH

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer2")


def touch_readout(E, half, min_control=8, use_touches=False, touch_max_dist=1.0,
                  max_contest=1.0, contested_min=0):
    """Per-frame team (0 home, 1 away, -1 unknown) from the event list."""
    n = len(half)
    lab = np.full(n, -1, np.int8)
    cur, last = -1, 0
    starts = []
    rec = E[E["kind"] == RECEIVE]
    for r in rec:
        f = int(r["frame"])
        dur = int(r["duration"])
        contested = r["contest"] > max_contest
        need = max(min_control, contested_min) if contested else min_control
        if dur >= need and r["team"] != cur:
            starts.append((f, int(r["team"])))
            cur = int(r["team"])
    if use_touches:
        for t in E[(E["kind"] == TOUCH) & (E["dist"] <= touch_max_dist)]:
            starts.append((int(t["frame"]), int(t["team"])))
        starts.sort()
    cur = -1
    pos = 0
    for f, t in starts:
        lab[pos:f] = cur
        cur, pos = t, f
    lab[pos:] = cur
    # restart each half from unknown, then back-fill the head of the half with its first team
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        seg = lab[idx]
        first = seg[seg >= 0][0] if (seg >= 0).any() else -1
        k = np.argmax(seg >= 0) if (seg >= 0).any() else len(seg)
        lab[idx[:k]] = first
    return lab


def baseline_labels():
    path = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
    import csv
    lab = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for _, l in r:
            lab.append({"H": 0, "A": 1}.get(l, -1))
    return np.array(lab, np.int8)


def turnovers(graded):
    z0 = load_layer("layer0")
    home, away = json.loads(str(z0["meta"]))["teams"]
    mapping, key = team_index_from_names(home, away)
    out = []
    for e in load_smart_tagging():
        t = mapping.get(key(e["team_name"]))
        if e["code"] == "ATTACKING TRANSITION" and t is not None and graded[e["start"]]:
            out.append((e["start"], t))
    return out


def score(lab, half, proxy, flips):
    g = half > 0
    m = g & (proxy != 0) & (lab >= 0)
    want = np.where(proxy > 0, 0, 1)
    agree = float(np.mean(lab[m] == want[m]))
    lags = []
    for T, t in flips:
        w = lab[T - 250:T + 250]
        # first frame from which the new team holds for at least 1 s
        hit = None
        for i in range(len(w) - 25):
            if (w[i:i + 25] == t).all():
                hit = i
                break
        if hit is not None:
            lags.append((hit - 250) / 25.0)
    # possessions: maximal runs of one team inside the graded halves
    runs = []
    for h in (1, 2):
        seg = lab[half == h]
        seg = seg[seg >= 0]
        if len(seg):
            ch = np.flatnonzero(np.diff(seg) != 0)
            bounds = np.r_[0, ch + 1, len(seg)]
            runs += list(np.diff(bounds) / 25.0)
    minutes = g.sum() / 1500.0
    lags = np.array(lags)
    return {"proxy_agreement": agree, "turnovers_found_within_10s": len(lags) / max(len(flips), 1),
            "turnover_lag_median_s": float(np.median(lags)) if len(lags) else None,
            "turnover_abs_lag_median_s": float(np.median(np.abs(lags))) if len(lags) else None,
            "possessions_per_min": len(runs) / minutes,
            "possession_s_median": float(np.median(runs)) if runs else None}


def main():
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    half = z["half"]
    proxy = F["st_possessor"]
    flips = turnovers(half > 0)
    E = L2["events"]
    R = {"n_turnovers": len(flips)}
    R["baseline_smooth50"] = score(baseline_labels(), half, proxy, flips)
    for mc in (8, 12, 20):
        R["touch_readout_min_control_{}".format(mc)] = score(
            touch_readout(E, half, min_control=mc), half, proxy, flips)
    for mx, cm in ((0.5, 20), (0.5, 40), (0.25, 25), (0.0, 25)):
        R["readout_contested>{}_needs_{}".format(mx, cm)] = score(
            touch_readout(E, half, min_control=8, max_contest=mx, contested_min=cm),
            half, proxy, flips)
    R["touch_readout_plus_close_touches"] = score(
        touch_readout(E, half, min_control=8, use_touches=True), half, proxy, flips)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "chain.md"), "w") as f:
        f.write("# Layers 0-2 together\n\n" + (__doc__ or "") + "\n```\n"
                + json.dumps(R, indent=1) + "\n```\n")
    for k, v in R.items():
        if isinstance(v, dict):
            print("{:38} agree {:.3f}  turnovers found {:.0%}  lag {:+.1f}s |{:.1f}s|  "
                  "poss/min {:.2f}  median {:.1f}s".format(
                      k, v["proxy_agreement"], v["turnovers_found_within_10s"],
                      v["turnover_lag_median_s"] or 0, v["turnover_abs_lag_median_s"] or 0,
                      v["possessions_per_min"], v["possession_s_median"] or 0))


if __name__ == "__main__":
    main()
