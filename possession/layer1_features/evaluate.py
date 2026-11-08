"""Evaluate layer 1 without any ground truth.

    python -m possession.layer1_features.evaluate     # reads cache/layer1_tactical.npz

The question for a feature is: does it carry evidence about who has the ball? The ground truth
is off-limits, so the yardstick is a proxy produced by an algorithm: Metrica's automatic Smart
Tagging, whose phase codes imply a side in possession (common.POSSESSOR). It is coarse (phases,
not possessions) and noisy, so absolute numbers understate the features; what it is good for is
comparing features and feature sets against each other.

  - per-feature AUC against the proxy, overall and split by whether the ball is visible
  - a combined logistic model with blocked (contiguous-time) cross-validation, by feature group
  - redundancy: feature pairs that are near-duplicates
  - dead-ball cues against the Smart Tagging set-piece windows
  - sanity ranges

Writes reports/layer1/report.md.
"""

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..common import REPORTS, ROOT, load_layer

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPORT_DIR = os.path.join(REPORTS, "layer1")

# features that are not evidence about possession (bookkeeping, or the proxy itself)
NOT_EVIDENCE = {"st_possessor", "st_set_piece", "st_set_piece_team", "half", "slot_h", "slot_a",
                "team_count_anomaly"}
PAIRS = ["relv", "closing", "nearest_fwd_speed", "n2", "n5", "dwell", "centroid_x",
         "spread_x", "spread_y", "def_line", "front_line", "centroid_fwd_speed", "mean_speed",
         "n_players", "ball_ahead", "area", "press", "n_opp_half", "fwd_share", "length"]


def oriented_auc(y, x):
    ok = ~np.isnan(x)
    if ok.sum() < 200 or len(np.unique(y[ok])) < 2:
        return np.nan, 0.0
    a = roc_auc_score(y[ok], x[ok])
    return max(a, 1 - a), float(ok.mean())


def add_differences(F):
    """Home-minus-away versions of every paired feature: possession evidence is relative."""
    D = {}
    for p in PAIRS:
        h, a = p + "_h", p + "_a"
        if h in F and a in F:
            D[p + "_diff"] = F[h].astype(float) - F[a].astype(float)
    return D


def design(F, names):
    """Matrix with NaN imputed to 0 after scaling, plus a missing indicator per feature."""
    cols, miss = [], []
    for n in names:
        x = F[n].astype(float)
        m = np.isnan(x)
        cols.append(np.where(m, np.nanmedian(x) if (~m).any() else 0, x))
        if m.any():
            miss.append(m.astype(float))
    X = np.column_stack(cols + miss)
    return X


def blocked_cv_auc(X, y, groups, n_folds=6):
    """Logistic regression, folds are contiguous blocks of time (no leakage across time)."""
    order = np.argsort(groups)
    folds = np.array_split(order, n_folds)
    scores = np.full(len(y), np.nan)
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        sc = StandardScaler().fit(X[train])
        m = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(X[train]), y[train])
        scores[test] = m.predict_proba(sc.transform(X[test]))[:, 1]
    return float(roc_auc_score(y, scores)), scores


def proxy_flips(graded):
    """Turnovers according to the Smart Tagging: the start of every ATTACKING TRANSITION phase,
    which is the moment its team won the ball. Returns [(frame, +1 home won / -1 away won)]."""
    import json as _json
    from ..common import load_smart_tagging, team_index_from_names
    z0 = load_layer("layer0")
    home, away = _json.loads(str(z0["meta"]))["teams"]
    mapping, key = team_index_from_names(home, away)
    flips = []
    for e in load_smart_tagging():
        t = mapping.get(key(e["team_name"]))
        if e["code"] == "ATTACKING TRANSITION" and t is not None and graded[e["start"]]:
            flips.append((e["start"], +1 if t == 0 else -1))
    return flips


def step_lag(score, flips, window=250):
    """For each hand-over, the time where a step from the old side to the new side fits the
    score best. Returns lags in seconds (+ = the score switches after the proxy)."""
    lags = []
    for T, new in flips:
        a, b = T - window, T + window
        if a < 0 or b > len(score):
            continue
        s = score[a:b] * new            # positive = agrees with the new possessor
        if np.isnan(s).mean() > 0.5:
            continue
        s = np.nan_to_num(s)
        pos, neg = np.sign(s) > 0, np.sign(s) < 0
        # agreement if the switch happens at k: old side before k, new side from k
        before = np.concatenate(([0], np.cumsum(neg)))
        after = np.concatenate(([0], np.cumsum(pos[::-1])))[::-1]
        k = int(np.argmax(before + after))
        lags.append((k - window) / 25.0)
    return np.array(lags)


def evaluate():
    F = {k: v for k, v in load_layer("layer1").items()}
    graded = F["half"] > 0
    proxy = F["st_possessor"]
    lab = graded & (proxy != 0)
    y = (proxy[lab] > 0).astype(int)          # 1 = home in possession per the proxy
    known = F["ball_known"][lab].astype(bool)
    R = {"proxy_frames": int(lab.sum()), "proxy_home_share": float(y.mean()),
         "proxy_coverage_of_graded": float(lab.sum() / graded.sum())}

    D = add_differences(F)
    allF = dict(F, **D)
    names = [k for k in allF if k not in NOT_EVIDENCE and allF[k].dtype != bool]
    rows = []
    for n in sorted(names):
        x = allF[n][lab].astype(float)
        auc, cov = oriented_auc(y, x)
        auc_k, _ = oriented_auc(y[known], x[known])
        auc_m, _ = oriented_auc(y[~known], x[~known])
        rows.append((n, auc, cov, auc_k, auc_m))
    rows.sort(key=lambda r: -np.nan_to_num(r[1]))
    R["per_feature"] = [{"feature": r[0], "auc": r[1], "coverage": r[2], "auc_ball_known": r[3],
                         "auc_ball_missing": r[4]} for r in rows]

    # feature groups for the combined model
    ball_now = ["margin", "d_h", "d_a", "relv_diff", "closing_diff", "n2_diff", "n5_diff",
                "dwell_diff", "nearest_fwd_speed_diff", "ball_x_home_frame", "ball_ahead_diff"]
    ball_multi = ball_now + ["ball_fwd_speed_home", "ball_fwd_1s", "margin_1s", "margin_3s", "margin_10s", "near_share_1s",
                             "near_share_3s", "near_share_10s", "margin_before_gap",
                             "margin_after_gap"]
    shape_full = ["territory", "line_gap", "centroid_x_diff", "spread_x_diff", "spread_y_diff",
                  "def_line_diff", "front_line_diff", "centroid_fwd_speed_diff",
                  "mean_speed_diff", "n_players_diff", "area_diff"]
    shape_lean = ["territory", "spread_x_diff", "spread_y_diff", "centroid_fwd_speed_diff",
                  "area_diff", "press_diff", "n_opp_half_diff", "fwd_share_diff", "length_diff"]
    groups = {"ball_now": ball_now, "ball_multiscale": ball_multi, "shape_full": shape_full,
              "shape_lean": shape_lean, "all": ball_multi + shape_full,
              "recommended": ball_multi + shape_lean}
    t = np.flatnonzero(lab)
    R["combined"] = {}
    for g, feats in groups.items():
        feats = [f for f in feats if f in allF]
        X = design({f: allF[f][lab] for f in feats}, feats)
        auc, scores = blocked_cv_auc(X, y, t)
        R["combined"][g] = {
            "auc": auc,
            "auc_ball_known": float(roc_auc_score(y[known], scores[known])),
            "auc_ball_missing": float(roc_auc_score(y[~known], scores[~known])),
            "n_features": len(feats)}

    # transition timing: how fast does the evidence switch sides at a proxy hand-over?
    flips = proxy_flips(graded)
    R["n_proxy_flips"] = len(flips)
    timing = {}
    for g in ("ball_multiscale", "shape_lean", "recommended"):
        feats = [f for f in groups[g] if f in allF]
        Xall = design({f: allF[f] for f in feats}, feats)
        sc = StandardScaler().fit(Xall[lab])
        m = LogisticRegression(max_iter=500).fit(sc.transform(Xall[lab]), y)
        score = m.decision_function(sc.transform(Xall))
        lg = step_lag(score, flips)
        timing["model_" + g] = lg
    for f in ("margin", "margin_1s", "near_share_3s", "ball_fwd_1s", "dwell_diff", "n2_diff",
              "area_diff", "territory", "spread_y_diff"):
        x = allF[f].astype(float)
        x = x - np.nanmedian(x[lab])
        lg = step_lag(x, flips)
        # orient: a feature whose AUC direction is reversed switches the other way
        auc = roc_auc_score(y[~np.isnan(allF[f][lab])], allF[f][lab][~np.isnan(allF[f][lab])])
        timing[f] = step_lag(x if auc >= 0.5 else -x, flips)
    R["transition_timing"] = {k: {"n": int(len(v)), "median_lag_s": float(np.median(v)),
                                  "median_abs_lag_s": float(np.median(np.abs(v))),
                                  "within_2s": float(np.mean(np.abs(v) <= 2))}
                              for k, v in timing.items() if len(v)}

    # redundancy among evidence features
    cand = [r[0] for r in rows if r[1] == r[1] and r[1] > 0.55]
    M = np.column_stack([allF[n][graded].astype(float) for n in cand])
    ok = ~np.isnan(M).any(axis=1)
    corr = np.corrcoef(M[ok].T) if ok.sum() > 100 else np.zeros((len(cand), len(cand)))
    red = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if abs(corr[i, j]) > 0.95:
                red.append((cand[i], cand[j], round(float(corr[i, j]), 3)))
    R["redundant_pairs"] = red

    # dead-ball cues vs set-piece windows
    sp = (F["st_set_piece"] > 0)[graded].astype(int)
    R["dead_ball_cues_vs_set_pieces"] = {}
    for n in ["mean_speed_all", "ball_speed", "ball_still_frames", "ball_outside", "ball_to_line",
              "ball_to_restart_spot", "ball_frames_since_seen", "last_seen_to_line"]:
        auc, cov = oriented_auc(sp, F[n][graded].astype(float))
        R["dead_ball_cues_vs_set_pieces"][n] = {"auc": auc, "coverage": cov}

    # sanity
    R["sanity"] = {
        "d_min": float(np.nanmin(F["d_h"])), "d_p99": float(np.nanpercentile(F["d_h"], 99)),
        "mean_speed_all_p50_p99": [float(np.nanpercentile(F["mean_speed_all"][graded], q))
                                   for q in (50, 99)],
        "n_players_h_max": int(F["n_players_h"][graded].max()),
        "ball_feature_coverage": float(F["ball_known"][graded].mean()),
        "shape_feature_coverage": float(np.mean(~np.isnan(F["centroid_x_h"][graded]))),
    }
    return R


def write_report(R):
    os.makedirs(REPORT_DIR, exist_ok=True)
    L = ["# Layer 1 evaluation", "",
         "Proxy: Smart Tagging phases imply the side in possession on {:.0%} of graded frames "
         "({} frames, home share {:.0%}).".format(R["proxy_coverage_of_graded"],
                                                   R["proxy_frames"], R["proxy_home_share"]),
         "", "## Combined (logistic, blocked CV)", "",
         "| group | features | AUC | ball visible | ball missing |", "|---|---|---|---|---|"]
    for g, v in R["combined"].items():
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
            g, v["n_features"], v["auc"], v["auc_ball_known"], v["auc_ball_missing"]))
    L += ["", "## Transition timing at {} proxy hand-overs".format(R["n_proxy_flips"]), "",
          "Where a step from the old side to the new side best fits the evidence, relative to the "
          "proxy's hand-over (+ = later). The proxy's own boundaries are only good to ~1-2 s.", "",
          "| evidence | median lag | median abs lag | within 2 s |", "|---|---|---|---|"]
    for k, v in R["transition_timing"].items():
        L.append("| {} | {:+.1f} s | {:.1f} s | {:.0%} |".format(
            k, v["median_lag_s"], v["median_abs_lag_s"], v["within_2s"]))
    L += ["", "## Per feature (oriented AUC vs proxy)", "",
          "| feature | AUC | coverage | ball visible | ball missing |", "|---|---|---|---|---|"]
    for r in R["per_feature"]:
        L.append("| {} | {:.3f} | {:.0%} | {} | {} |".format(
            r["feature"], np.nan_to_num(r["auc"]), r["coverage"],
            "-" if r["auc_ball_known"] != r["auc_ball_known"] else "{:.3f}".format(r["auc_ball_known"]),
            "-" if r["auc_ball_missing"] != r["auc_ball_missing"] else "{:.3f}".format(r["auc_ball_missing"])))
    L += ["", "## Redundant pairs (|r| > 0.95)", ""] + \
         ["- {} ~ {} (r = {})".format(*p) for p in R["redundant_pairs"]] + \
         ["", "## Dead-ball cues vs set-piece windows", "", "```",
          json.dumps(R["dead_ball_cues_vs_set_pieces"], indent=1), "```",
          "", "## Sanity", "", "```", json.dumps(R["sanity"], indent=1), "```"]
    with open(os.path.join(REPORT_DIR, "report.md"), "w") as f:
        f.write("\n".join(L))


def main():
    R = evaluate()
    write_report(R)
    for g, v in R["combined"].items():
        print("combined {:6} AUC {:.3f}  (ball visible {:.3f}, missing {:.3f})".format(
            g, v["auc"], v["auc_ball_known"], v["auc_ball_missing"]))
    print("proxy hand-overs:", R["n_proxy_flips"])
    for k, v in R["transition_timing"].items():
        print("  timing {:26} median lag {:+.1f}s  |lag| {:.1f}s  within 2s {:.0%}".format(
            k, v["median_lag_s"], v["median_abs_lag_s"], v["within_2s"]))
    print("top features:")
    for r in R["per_feature"][:15]:
        print("  {:28} {:.3f}  known {}  missing {}".format(
            r["feature"], r["auc"], round(r["auc_ball_known"], 3) if r["auc_ball_known"] == r["auc_ball_known"] else "-",
            round(r["auc_ball_missing"], 3) if r["auc_ball_missing"] == r["auc_ball_missing"] else "-"))
    print("redundant:", R["redundant_pairs"])
    print("dead-ball cues:", {k: round(v["auc"], 3) for k, v in R["dead_ball_cues_vs_set_pieces"].items()})
    print("sanity:", R["sanity"])
    print("report:", os.path.relpath(os.path.join(REPORT_DIR, "report.md"), ROOT))


if __name__ == "__main__":
    main()
