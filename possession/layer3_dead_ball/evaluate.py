"""Evaluate layer 3 without any ground truth.

    python -m possession.layer3_dead_ball.evaluate          # reads cache/layer3_tactical.npz
    python -m possession.layer3_dead_ball.evaluate --sweep  # also sweeps the main weights

Checks, each from something other than the cues being scored where possible:

  known stoppages     the 2nd half is graded from frame 110,808 but kicks off at 113,742
                      (FAQ.md), so those frames are dead; so is the minute before each restart
                      after a goal (Smart Tagging "Kick Off (After Goal)").
  set-piece restarts  every Smart Tagging set piece (throw-in, corner, goal kick, free kick,
                      kick-off) is a restart: a dead -> alive switch should fall inside its
                      window. (The windows are also a weak cue, so this is not fully independent.)
  playing time        LaLiga's average effective playing time is ~53.4 min a match (ESPN,
                      2022-23); over ~106 graded minutes that is ~50% dead. Single matches vary
                      by several minutes either way, so 45-60 min of play (43-57% dead) is the
                      plausible band.
  stoppages           a match has ~100-130 stoppages (throw-ins, goal kicks, corners, fouls,
                      goals, subs), most lasting 5-40 s.
  physics             while the ball is dead nobody needs to run: player speed during detected
                      dead vs alive.

Writes reports/layer3/report.md.
"""

import argparse
import json
import os
import warnings

import numpy as np

from .. import config as C
from ..common import REPORTS, ROOT, load_layer, load_smart_tagging
from .deadball import detect

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer3")
SECOND_HALF_GRADED, SECOND_HALF_KICKOFF = 110808, 113742


def runs(mask):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def evaluate_state(dead, z, F, tags):
    half = z["half"]
    g = half > 0
    R = {}
    R["dead_share"] = float(dead[g].mean())
    R["dead_minutes"] = float(dead[g].sum() / 1500)
    R["alive_minutes"] = float((~dead.astype(bool) & g).sum() / 1500)
    # stoppages
    a, b = runs(dead.astype(bool) & g)
    dur = (b - a) / 25.0
    R["stoppages"] = int(len(a))
    R["stoppage_s_p25_p50_p75"] = [float(np.percentile(dur, q)) for q in (25, 50, 75)]
    R["stoppages_under_3s"] = int(np.sum(dur < 3))
    al, bl = runs(~dead.astype(bool) & g)
    R["alive_spells_under_3s"] = int(np.sum((bl - al) < 75))
    # known stoppages
    pre = slice(SECOND_HALF_GRADED, SECOND_HALF_KICKOFF)
    R["second_half_prekickoff_dead"] = float(dead[pre].mean())
    goals = [t for t in tags if t["code"] == "SET PIECES" and t["type"] == "Kick Off (After Goal)"]
    R["before_goal_kickoffs_dead"] = float(np.mean([dead[t["start"] - 1500:t["start"]].mean()
                                                    for t in goals])) if goals else None
    # set-piece restarts
    sps = [t for t in tags if t["code"] == "SET PIECES" and g[min(t["start"], len(g) - 1)]]
    hits, by_type = [], {}
    for t in sps:
        s, e = t["start"], t["end"]
        w = dead[max(0, s - 50):e + 50]
        hit = bool(np.any(np.diff(w.astype(int)) == -1))   # a dead -> alive switch
        hits.append(hit)
        by_type.setdefault(t["type"], []).append(hit)
    R["set_piece_restarts_found"] = float(np.mean(hits)) if hits else None
    R["set_piece_restarts_by_type"] = {k: [int(sum(v)), len(v)] for k, v in by_type.items()}
    # certain: ball well beyond a line
    out = np.nan_to_num(F["ball_outside"].astype(float))
    cert = g & (out > 1.5)
    R["ball_far_out_dead"] = float(dead[cert].mean()) if cert.any() else None
    # physics
    ms = F["mean_speed_all"].astype(float)
    R["player_speed_dead_vs_alive"] = [float(np.nanmedian(ms[g & (dead == 1)])),
                                       float(np.nanmedian(ms[g & (dead == 0)]))]
    return R


def checks(R):
    c = []

    def add(name, value, ok, target):
        c.append((name, value, bool(ok), target))
    add("2nd-half pre-kickoff is dead", round(R["second_half_prekickoff_dead"], 3),
        R["second_half_prekickoff_dead"] > 0.95, "> 0.95")
    if R["before_goal_kickoffs_dead"] is not None:
        add("minute before each after-goal kick-off is dead", round(R["before_goal_kickoffs_dead"], 3),
            R["before_goal_kickoffs_dead"] > 0.85, "> 0.85")
    add("ball > 1.5 m beyond a line is dead", round(R["ball_far_out_dead"], 3),
        R["ball_far_out_dead"] > 0.97, "> 0.97")
    add("set-piece windows contain a restart", round(R["set_piece_restarts_found"], 3),
        R["set_piece_restarts_found"] >= 0.85, ">= 0.85")
    add("effective playing time (min; LaLiga avg ~53.4)", round(R["alive_minutes"], 1),
        45 <= R["alive_minutes"] <= 60, "45-60")
    add("number of stoppages", R["stoppages"], 90 <= R["stoppages"] <= 150, "90-150")
    add("stoppages shorter than 3 s (implausible)", R["stoppages_under_3s"],
        R["stoppages_under_3s"] <= 10, "<= 10")
    add("players slower when dead (median m/s, dead vs alive)",
        [round(x, 2) for x in R["player_speed_dead_vs_alive"]],
        R["player_speed_dead_vs_alive"][0] < 0.75 * R["player_speed_dead_vs_alive"][1],
        "dead < 0.75 x alive")
    return c


def sweep(z, F, L2, tags):
    grid = {"DEAD_SWITCH_COST": [4.0, 8.0, 16.0], "DEAD_W_SPEED": [0.75, 1.5, 3.0],
            "DEAD_SPEED_PIVOT": [1.3, 1.6, 1.9], "DEAD_W_GAP_LONG": [0.75, 1.5, 3.0],
            "DEAD_W_SET_PIECE": [0.0, 1.0, 2.0], "DEAD_W_STILL": [0.0, 1.5, 3.0],
            "DEAD_W_KICKOFF_SHAPE": [0.0, 3.0], "DEAD_MIN_DEAD_S": [0.0, 3.0, 5.0]}
    rows = []
    for name, vals in grid.items():
        old = getattr(C, name)
        for v in vals:
            setattr(C, name, v)
            dead, _, _ = detect(z, F, L2)
            R = evaluate_state(dead, z, F, tags)
            n_ok = sum(x[2] for x in checks(R))
            rows.append((name, v, n_ok, R["dead_share"], R["stoppages"], R["stoppages_under_3s"],
                         R["set_piece_restarts_found"], R["before_goal_kickoffs_dead"],
                         R["second_half_prekickoff_dead"]))
        setattr(C, name, old)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    tags = load_smart_tagging()
    dead = load_layer("layer3")["dead"]
    R = evaluate_state(dead, z, F, tags)
    C_ = checks(R)
    lines = ["# Layer 3 evaluation", "", "| check | value | target | |", "|---|---|---|---|"]
    for name, value, ok, target in C_:
        lines.append("| {} | {} | {} | {} |".format(name, value, target, "PASS" if ok else "FAIL"))
        print("{:5}  {:<56} {}".format("PASS" if ok else "FAIL", name, value))
    lines += ["", "```", json.dumps(R, indent=1), "```"]
    if args.sweep:
        rows = sweep(z, F, L2, tags)
        lines += ["", "## Sweep", "", "| param | value | checks passed | dead share | stoppages | "
                  "< 3 s | set pieces | goal | pre-kickoff |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append("| {} | {} | {} | {:.3f} | {} | {} | {:.2f} | {:.2f} | {:.2f} |".format(*r))
            print("{:20} {:6} ok {}  dead {:.3f}  stoppages {:4}  <3s {:3}  setpieces {:.2f}  "
                  "goal {:.2f}  prekick {:.2f}".format(*r))
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print(json.dumps({k: R[k] for k in ("stoppage_s_p25_p50_p75", "set_piece_restarts_by_type",
                                         "alive_spells_under_3s", "dead_minutes")}))


if __name__ == "__main__":
    main()
