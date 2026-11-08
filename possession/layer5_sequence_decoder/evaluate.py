"""Evaluate layer 5 without any ground truth (and the public window as a check only).

    python -m possession.layer5_sequence_decoder.evaluate          # current settings
    python -m possession.layer5_sequence_decoder.evaluate --sweep  # sweep the main settings

Selection uses only:
  - agreement with the Smart Tagging proxy (all live frames, and stale ones)
  - Smart Tagging turnovers found within 10 s, and their lag
  - consistency: certain-possession frames (layer 4 labels) decoded to the same team
  - reception consistency: share of uncontested receptions (layer 2) whose team holds the ball
    while he controls it. The challenge counts any controlled possession, however brief: the
    public window (CHALLENGE.md: "use it to understand how these cases are labelled") has
    possessions of 0.8 s and 1.3 s - a reception by the other team is a possession.
  - kick consistency: share of kicks (layer 2: a player plays the ball away first time) whose
    team has the ball right after - the team that last played the ball keeps it until the other
    team controls it, which is how the public window labels clearances and first-time passes
  - possessions per minute of play (sanity: not below 2)
The public 5 minutes is printed next to them but never used to choose settings.
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np

from .. import config as C
from .. import compare_models as cm
from ..common import REPORTS, load_layer
from ..layer2_touches.evaluate_chain import turnovers
from .decoder import decode

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer5")


def metrics(lab, z, F, L2, L4, flips):
    half = z["half"]
    proxy = F["st_possessor"]
    px = cm.proxy_scores(lab, half, proxy, flips)
    alive = (lab < 2) & (half > 0)
    m = alive & (proxy != 0)
    known = z["ball_gap_len"] == 0
    since = L2["frames_since_control"]
    stale = m & ((~known) | (since > 250) | (since < 0))
    y = L4["label"]
    cert = (y >= 0) & alive
    E = L2["events"]
    rec = E[(E["kind"] == 0) & (E["contest"] <= 0.25)]
    honoured = [np.mean(lab[int(r["frame"]):int(r["frame"] + r["duration"])] == r["team"]) > 0.5
                for r in rec if lab[int(r["frame"])] < 2]
    kk = E[E["kind"] == 8]
    kicks_honoured = [lab[min(len(lab) - 1, int(k["frame"]) + 5)] == k["team"]
                      for k in kk if lab[int(k["frame"])] < 2]
    runs = []
    for h in (1, 2):
        seg = lab[half == h]
        b = np.flatnonzero(np.diff(seg) != 0) + 1
        for a, e in zip(np.r_[0, b], np.r_[b, len(seg)]):
            if seg[a] < 2:
                runs.append(e - a)
    runs = np.array(runs) / 25.0
    pub = cm.public_scores(cm.to_letters(lab))
    return {"proxy_agreement": px["proxy_agreement"],
            "proxy_agreement_stale": float(np.mean((lab[stale] == 0) == (proxy[stale] > 0))),
            "turnovers_found": px["turnovers_found_within_10s"],
            "turnover_abs_lag_s": px["turnover_abs_lag_median_s"],
            "certain_consistency": float(np.mean((lab[cert] == 0) == (y[cert] == 1))),
            "receptions_honoured": float(np.mean(honoured)),
            "kicks_honoured": float(np.mean(kicks_honoured)) if kicks_honoured else 1.0,
            "possessions_per_min": px["possessions_per_min"],
            "possession_s_median": float(np.median(runs)),
            "short_possessions_share": float(np.mean(runs < 2)),
            "public_spellF1_check_only": pub["spellF1"], "public_macroF1_check_only": pub["macroF1"],
            "public_possessions_check_only": pub["possessions"]}


def objective(r):
    """Higher is better. No ground truth involved: Smart Tagging agreement and turnovers, plus
    honouring receptions (the challenge's definition of possession)."""
    rate = r["possessions_per_min"]
    rate_pen = max(0.0, 2.0 - rate)
    return (r["proxy_agreement"] + 0.5 * r["proxy_agreement_stale"] + 0.5 * r["turnovers_found"]
            + r["certain_consistency"] + 2.0 * r["receptions_honoured"] + 1.0 * r["kicks_honoured"]
            - 0.5 * rate_pen)


def sweep(z, F, L2, L3, L4, flips):
    grid = {"SEQ_W_PLAYERS": [0.01, 0.02, 0.05],
            "SEQ_SWITCH_COST": [2.0, 4.0],
            "SEQ_W_RECEIVE": [4.0, 8.0],
            "SEQ_W_KICK": [3.0, 6.0, 12.0, 25.0],
            "SEQ_W_DUR": [0.3, 1.0]}
    rows = []
    keys = list(grid)
    old = {k: getattr(C, k) for k in keys}
    for vals in itertools.product(*grid.values()):
        for k, v in zip(keys, vals):
            setattr(C, k, v)
        r = metrics(decode(z, L2, L3, L4), z, F, L2, L4, flips)
        rows.append((dict(zip(keys, vals)), objective(r), r))
    for k, v in old.items():
        setattr(C, k, v)
    rows.sort(key=lambda x: -x[1])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    z, F, L2, L3, L4 = (load_layer(x) for x in ("layer0", "layer1", "layer2", "layer3", "layer4"))
    flips = turnovers(z["half"] > 0)
    lab = load_layer("layer5")["label"]
    R = {"current": metrics(lab, z, F, L2, L4, flips)}
    R["current"]["objective"] = objective(R["current"])
    print(json.dumps(R["current"], indent=1))
    if args.sweep:
        rows = sweep(z, F, L2, L3, L4, flips)
        R["sweep_top10"] = [(p, o, r) for p, o, r in rows[:10]]
        for p, o, r in rows[:12]:
            print("{:.3f} {}  agree {:.3f} stale {:.3f} turn {:.2f} cert {:.3f} poss/min {:.2f} "
                  "recv {:.3f} kick {:.2f} | public spellF1 {:.3f} poss {}".format(
                      o, p, r["proxy_agreement"], r["proxy_agreement_stale"], r["turnovers_found"],
                      r["certain_consistency"], r["possessions_per_min"], r["receptions_honoured"],
                      r["kicks_honoured"],
                      r["public_spellF1_check_only"], r["public_possessions_check_only"]))
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "report.md"), "w") as f:
        f.write("# Layer 5 evaluation\n\n" + __doc__ + "\n```\n" + json.dumps(R, indent=1, default=str) + "\n```\n")


if __name__ == "__main__":
    main()
