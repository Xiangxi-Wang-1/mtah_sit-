"""Run the pipeline on the bench games and score it with the official scorer.

    python -m possession.bench.score                 # run layers 1-5 on bench1-3, then score
    python -m possession.bench.score --no-run        # score what is in the cache
    python -m possession.bench.score --label v1      # also append the result to reports/bench/history.md

Per game: spellF1 and macroF1 (official evaluation/possession_metrics.py), alive/dead F1, team
accuracy on alive frames, possession count. Two diagnostic swaps say which half of the problem
costs the most:
  our team + true dead    the pipeline's team calls, with the answer key's stoppages
  true team + our dead    the answer key's team calls, with the pipeline's stoppages

Nothing here is tuned on the challenge match; this is the score the pipeline is chosen on.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from ..common import CACHE, REPORTS, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

BENCH = os.path.join(CACHE, "bench")
LAYERS = ["layer1_features", "layer2_touches", "layer3_dead_ball", "layer4_possession_model",
          "layer5_sequence_decoder"]


def run_layers(tag, layers=LAYERS):
    env = dict(os.environ, POSSESSION_GAME=tag)
    for layer in layers:
        r = subprocess.run([sys.executable, "-m", "possession.{}.run".format(layer)], cwd=ROOT,
                           env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("{} on {}:\n{}".format(layer, tag, r.stderr[-2000:]))


def read_letters(path):
    with open(path) as f:
        f.readline()
        return np.array([line.rstrip("\n").split(",")[1] for line in f])


def metrics(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "alive_dead_F1": r["alive_dead_f1"], "team_acc_alive": r["team_acc_both_alive"],
            "possessions": r["n_pred_spells"], "truth_possessions": r["n_gt_spells"]}


def score_game(tag):
    gt = read_letters(os.path.join(BENCH, tag + "_gt.csv"))
    base = read_letters(os.path.join(BENCH, tag + "_baseline.csv"))
    L5 = np.load(os.path.join(CACHE, "layer5_{}.npz".format(tag)))
    lab = L5["label"]
    ours = np.where(lab == 0, "H", np.where(lab == 1, "A", "D"))
    ours = np.where(gt == "X", "X", ours)
    # diagnostic swaps
    team = L5["team"] if "team" in L5.files else None
    our_team_true_dead = np.where(gt == "D", "D", np.where(ours == "D", _fill_team(ours), ours))
    true_team_our_dead = np.where(ours == "D", "D", np.where(gt == "D", _fill_team(gt), gt))
    our_team_true_dead = np.where(gt == "X", "X", our_team_true_dead)
    true_team_our_dead = np.where(gt == "X", "X", true_team_our_dead)
    del team
    return {"baseline": metrics(gt, base), "pipeline": metrics(gt, ours),
            "our team + true dead": metrics(gt, our_team_true_dead),
            "true team + our dead": metrics(gt, true_team_our_dead)}


def _fill_team(letters):
    """Where a label is D, carry the last H/A forward (then backward) - a side for every frame."""
    s = np.where(np.isin(letters, ["H", "A"]), letters, "")
    idx = np.where(s != "", np.arange(len(s)), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, s[np.clip(idx, 0, None)], "")
    first = np.flatnonzero(s != "")
    out[out == ""] = s[first[0]] if len(first) else "H"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--label", default=None)
    a = p.parse_args()
    R = {}
    tags = ["bench{}".format(g) for g in a.games]
    if not a.no_run:
        # layers 1-3 everywhere first: layer 4 of each game trains on the other games
        for tag in tags:
            run_layers(tag, LAYERS[:3])
        for tag in tags:
            run_layers(tag, LAYERS[3:])
    for tag in tags:
        t0 = time.time()
        R[tag] = score_game(tag)
        print(tag, "{:.0f}s".format(time.time() - t0), json.dumps(
            {k: {m: round(v, 3) for m, v in d.items()} for k, d in R[tag].items()}))
    models = list(next(iter(R.values())))
    keys = ["spellF1", "macroF1", "alive_dead_F1", "team_acc_alive"]
    mean = {m: {k: float(np.mean([R[t][m][k] for t in R])) for k in keys} for m in models}
    lines = ["| | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for k in keys:
        lines.append("| {} (mean of {}) | ".format(k, len(R)) +
                     " | ".join("{:.3f}".format(mean[m][k]) for m in models) + " |")
    for t in R:
        lines.append("| spellF1 {} | ".format(t) +
                     " | ".join("{:.3f}".format(R[t][m]["spellF1"]) for m in models) + " |")
        lines.append("| possessions {} (truth {}) | ".format(t, R[t]["pipeline"]["truth_possessions"]) +
                     " | ".join(str(R[t][m]["possessions"]) for m in models) + " |")
    table = "\n".join(lines)
    print(table)
    out = os.path.join(REPORTS, "bench")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("# Bench score\n\n" + __doc__ + "\n" + table + "\n\n```\n" + json.dumps(R, indent=1)
                + "\n```\n")
    if a.label:
        with open(os.path.join(out, "history.md"), "a") as f:
            f.write("{}  spellF1 {:.3f}  macroF1 {:.3f}  alive/dead {:.3f}  team {:.3f}\n".format(
                a.label, mean["pipeline"]["spellF1"], mean["pipeline"]["macroF1"],
                mean["pipeline"]["alive_dead_F1"], mean["pipeline"]["team_acc_alive"]))


if __name__ == "__main__":
    main()
