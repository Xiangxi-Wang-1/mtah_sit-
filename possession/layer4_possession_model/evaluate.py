"""Evaluate layer 4 without any ground truth.

    python -m possession.layer4_possession_model.evaluate            # reads cache/layer4

  held-out certain moments  out-of-fold accuracy / AUC on the certain-possession frames, per
                            frame and per control spell, and calibration (Brier score)
  Smart Tagging proxy       AUC on the frames a Smart Tagging phase implies a side, split by
                            whether the ball is seen, and on "stale" frames (ball missing, or no
                            control for > 10 s) - where layers 0-3 are weakest
  complement                where the layers 0-3 read-out disagrees with the proxy, how often
                            the model agrees with it
  timing                    lag at the 24 Smart Tagging turnovers

Writes reports/layer4/report.md.
"""

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from ..common import REPORTS, load_layer
from ..layer1_features.evaluate import proxy_flips, step_lag
from ..layer2_touches.evaluate_chain import touch_readout

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer4")


# Players-view variants compared during development (out-of-fold, same certain frames):
# held-out certain AUC / accuracy | Smart Tagging AUC | accuracy on stale frames vs Smart Tagging
VARIANTS = {
    "moment model, leaves 15":                [0.943, 0.906, 0.940, 0.886],
    "labels without passes":                  [0.934, 0.893, 0.942, 0.835],
    "moment model, leaves 31":                [0.945, 0.910, 0.943, 0.888],
    "context model (30 s, +/-5 s), leaves 31": [0.966, 0.938, 0.933, 0.856],
    "blend 50/50 (chosen)":                   [0.964, 0.927, 0.944, 0.893],
}


def auc(y, p):
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok])) if ok.sum() > 100 and len(np.unique(y[ok])) > 1 else None


def evaluate(L4, z, F, L2, L3):
    y, seg = L4["label"], L4["label_segment"]
    half = z["half"]
    alive = (L3["dead"] == 0) & (half > 0)
    lab = y >= 0
    proxy = F["st_possessor"]
    pm = alive & (proxy != 0)
    py = (proxy[pm] > 0).astype(int)
    known = (z["ball_gap_len"] == 0)
    since = L2["frames_since_control"]
    stale = (~known) | (since > 250) | (since < 0)
    team = touch_readout(L2["events"], half, min_control=8, max_contest=0.25, contested_min=25)
    readout_home = (team == 0).astype(int)
    wrong = pm & (readout_home != (proxy > 0))
    R = {"labelled_frames": int(lab.sum()), "labelled_spells": int(len(np.unique(seg[lab]))),
         "readout_proxy_agreement": float(np.mean(readout_home[pm] == py)),
         "readout_proxy_agreement_stale": float(np.mean(readout_home[pm & stale] == (proxy[pm & stale] > 0)))}
    flips = proxy_flips(half > 0)
    for view in ("players", "full"):
        p = L4["p_home_" + view].astype(float)
        yy = y[lab]
        pp = p[lab]
        spells = {}
        for s_, v, t in zip(seg[lab], pp, yy):
            spells.setdefault(s_, [[], t])[0].append(v)
        spell_acc = float(np.mean([(np.mean(v) > 0.5) == (t == 1) for v, t in spells.values()]))
        r = {"oof_auc_certain": auc(yy, pp), "oof_acc_certain": float(np.mean((pp > 0.5) == (yy == 1))),
             "oof_spell_acc": spell_acc, "brier": float(np.mean((pp - yy) ** 2)),
             "proxy_auc": auc(py, p[pm]), "proxy_auc_ball_seen": auc((proxy[pm & known] > 0).astype(int), p[pm & known]),
             "proxy_auc_ball_missing": auc((proxy[pm & ~known] > 0).astype(int), p[pm & ~known]),
             "proxy_auc_stale": auc((proxy[pm & stale] > 0).astype(int), p[pm & stale]),
             "proxy_acc": float(np.mean((p[pm] > 0.5) == (py == 1))),
             "proxy_acc_stale": float(np.mean((p[pm & stale] > 0.5) == (proxy[pm & stale] > 0))),
             "right_where_readout_wrong": float(np.mean((p[wrong] > 0.5) == (proxy[wrong] > 0)))}
        lg = step_lag(np.nan_to_num(p - 0.5), flips)
        r["turnover_lag_median_s"] = float(np.median(lg))
        r["turnover_abs_lag_median_s"] = float(np.median(np.abs(lg)))
        R[view] = r
    return R


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    z, F = load_layer("layer0"), load_layer("layer1")
    L2, L3, L4 = load_layer("layer2"), load_layer("layer3"), load_layer("layer4")
    R = evaluate(L4, z, F, L2, L3)
    print(json.dumps(R, indent=1))
    R["variant_comparison"] = VARIANTS
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "report.md"), "w") as f:
        f.write("# Layer 4 evaluation\n\n" + __doc__ + "\n```\n" + json.dumps(R, indent=1, default=str)
                + "\n```\n")


if __name__ == "__main__":
    main()
