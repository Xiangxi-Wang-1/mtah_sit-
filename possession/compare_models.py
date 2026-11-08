"""Compare the official baseline with our read-outs, model by model.

    python -m possession.compare_models

Models
  baseline     the official baseline_smooth50 (nearest player, +/- 2 s majority vote)
  layers 0-2   touch read-out (possession changes only when the other team controls the ball;
               contested control must last 1 s); D wherever the ball is missing
  layers 0-5 v0  the full pipeline before the bench: event-anchored semi-Markov decoder (layer 5)
               over receptions, kicks and the self-trained possession model, stoppages from layer 3
  layers 0-5 v1  the same decoder over the cross-game models (layer 4, trained on the three bench
               games with true labels): P(home has the ball) per frame and learned change points;
               decoder settings chosen on the bench
  layers 0-3   the same possession read-out, with stoppages from layer 3: D during each dead
               spell, and after each restart the side is the first team to control the ball

Scored two ways:
  public 5 min  the official scorer against the public ground truth. This is a correctness
                check, not a tuning set (RULES.md): nothing in the pipeline is chosen from it.
                17 possessions, so one possession moves spellF1 by ~6 points.
  full match    agreement with the Smart Tagging proxy on H/A frames, the Smart Tagging
                turnovers, and possessions per minute (football: ~2-4).

Writes reports/model_comparison.md.
"""

import csv
import json
import os
import sys
import warnings

import numpy as np

from .common import REPORTS, ROOT, load_layer
from .layer2_touches.evaluate_chain import baseline_labels, score, touch_readout, turnovers
from .layer2_touches.touches import RECEIVE

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def with_dead(team, dead, E, half):
    """Team per frame with stoppages: D (=2) while dead; after a restart, the first team to
    control the ball has it from the restart on."""
    lab = team.astype(np.int8).copy()
    rec = E[E["kind"] == RECEIVE]
    rf = rec["frame"]
    restarts = np.flatnonzero((dead[:-1] == 1) & (dead[1:] == 0)) + 1
    for r in restarts:
        nxt = rec[(rf >= r)]
        if not len(nxt):
            continue
        f = int(nxt[0]["frame"])
        # only within the same spell of play
        if dead[r:f].any():
            continue
        lab[r:f] = nxt[0]["team"]
    lab[dead == 1] = 2
    return lab


def to_letters(lab):
    return np.where(lab == 0, "H", np.where(lab == 1, "A", np.where(lab == 2, "D", "X")))


def public_scores(letters):
    gt = pm.label_codes(pm.read_label_csv(os.path.join(
        ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    g = np.array([l for _, l in csv.reader(open(os.path.join(
        ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))][1:])
    r = pm.evaluate(gt, pm.label_codes(letters))
    ha = np.isin(g, ["H", "A"])
    dd = g == "D"
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "team_right_on_HA_frames": float(np.mean(letters[ha] == g[ha])),
            "dead_right_on_D_frames": float(np.mean(letters[dd] == "D")),
            "possessions": int(r["n_pred_spells"]), "mean_possession_s": r["mean_spell_duration_s"]}


def proxy_scores(lab, half, proxy, flips):
    team_only = np.where(lab == 2, -1, lab)   # dead frames have no side
    s = score(team_only, half, proxy, flips)
    # possessions with stoppages splitting them
    runs = []
    for h in (1, 2):
        seg = lab[half == h]
        b = np.flatnonzero(np.diff(seg) != 0) + 1
        for a, e in zip(np.r_[0, b], np.r_[b, len(seg)]):
            if seg[a] in (0, 1):
                runs.append((e - a) / 25.0)
    minutes = (half > 0).sum() / 1500.0
    s["possessions_per_min"] = len(runs) / minutes
    s["possession_s_median"] = float(np.median(runs))
    s["dead_share"] = float(np.mean(lab[half > 0] == 2))
    return s


def main():
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    L3 = load_layer("layer3")
    half = z["half"]
    E = L2["events"]
    proxy = F["st_possessor"]
    flips = turnovers(half > 0)
    known = z["ball_gap_len"] == 0

    team = touch_readout(E, half, min_control=8, max_contest=0.25, contested_min=25)
    base = baseline_labels()
    base = np.where(base < 0, 2, base)                       # the baseline's D is ball-missing
    prev = np.where(known, team, 2).astype(np.int8)
    cur = with_dead(team, L3["dead"], E, half)

    L5 = load_layer("layer5")
    # layers 0-5 before the bench: self-trained players model only, no change points
    from . import config as C
    from .layer5_sequence_decoder.decoder import decode
    keep = (C.SEQ_W_PLAYERS, C.SEQ_W_FULL, C.SEQ_W_CP)
    C.SEQ_W_PLAYERS, C.SEQ_W_FULL, C.SEQ_W_CP = 0.02, 0.0, 0.0
    v0 = decode(z, L2, L3, load_layer("layer4"))
    C.SEQ_W_PLAYERS, C.SEQ_W_FULL, C.SEQ_W_CP = keep
    models = {"baseline (smooth50)": base, "layers 0-2": prev, "layers 0-3": cur,
              "layers 0-5 v0": v0, "layers 0-5 v1 (current)": L5["label"].astype(np.int8)}
    R = {}
    for name, lab in models.items():
        R[name] = {"public_5min": public_scores(to_letters(lab)),
                   "full_match_proxy": proxy_scores(lab, half, proxy, flips)}

    rows = [("spellF1 (ranked on)", "public_5min", "spellF1", "{:.3f}"),
            ("macroF1 (tie-break)", "public_5min", "macroF1", "{:.3f}"),
            ("H/A frames with right team", "public_5min", "team_right_on_HA_frames", "{:.1%}"),
            ("D frames called dead", "public_5min", "dead_right_on_D_frames", "{:.1%}"),
            ("possessions (truth 17)", "public_5min", "possessions", "{}"),
            ("mean possession s (truth 14.2)", "public_5min", "mean_possession_s", "{:.1f}"),
            ("Smart Tagging agreement", "full_match_proxy", "proxy_agreement", "{:.3f}"),
            ("turnovers found (of 24)", "full_match_proxy", "turnovers_found_within_10s", "{:.0%}"),
            ("possessions per min (~2-4)", "full_match_proxy", "possessions_per_min", "{:.2f}"),
            ("dead share (~50%)", "full_match_proxy", "dead_share", "{:.1%}")]
    names = list(models)
    lines = ["| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, part, key, fmt in rows:
        lines.append("| {} | ".format(label) + " | ".join(fmt.format(R[n][part][key])
                                                         for n in names) + " |")
    table = "\n".join(lines)
    print(table)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "model_comparison.md"), "w") as f:
        f.write("# Model comparison\n\n" + __doc__ + "\n" + table + "\n\n```\n"
                + json.dumps(R, indent=1) + "\n```\n")


if __name__ == "__main__":
    main()
