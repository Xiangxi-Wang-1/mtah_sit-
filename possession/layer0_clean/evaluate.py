"""Evaluate layer 0 without any ground truth.

    python -m possession.layer0_clean.evaluate        # reads cache/layer0_tactical.npz

Every check here is self-supervised: physics (nothing moves faster than it can), agreement with
the ATD's own speed column, and hiding ball samples we do have to see how well a gap fill would
have recovered them. Writes reports/layer0/report.md plus figures.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config as C  # noqa: E402
from .ball import MISSING, FILLED, OBSERVED, REMOVED, fill_gaps, remove_false_detections  # noqa
from .run import DEFAULT_OUT, ROOT  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "layer0")


def _runs(mask):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def gap_lengths(known, graded):
    """Lengths of runs of unknown ball inside graded frames."""
    a, b = _runs(~known & graded)
    return b - a


def implied_speed(xy):
    """Speed between consecutive frames where both are known, m/s."""
    step = np.hypot(*(xy[1:] - xy[:-1]).T) / C.DT
    return step[~np.isnan(step)]


def gap_fill_benchmark(xy_raw, half, removed, lengths=(1, 2, 3, 5, 8, 10, 12, 15, 20, 25), samples=1500, seed=0):
    """Hide L observed frames inside long observed runs, fill, and measure the error (m)."""
    rng = np.random.default_rng(seed)
    xy = xy_raw.copy()
    xy[removed] = np.nan
    obs = ~np.isnan(xy[:, 0])
    a, b = _runs(obs)
    out = {}
    for L in lengths:
        need = L + 8  # 4 frames of context either side for the endpoint velocity
        ok = np.flatnonzero((b - a) >= need)
        if ok.size == 0:
            continue
        errs = {"linear": [], "hermite": []}
        picks = rng.choice(ok, size=min(samples, ok.size * 4), replace=True)
        # Hide one window per pick; batch many hides into one array where they do not overlap.
        for start_run in np.array_split(picks, max(1, len(picks) // 200)):
            test = xy.copy()
            hidden = []
            used = set()
            for r in start_run:
                if r in used:
                    continue
                used.add(r)
                s = rng.integers(a[r] + 4, b[r] - 4 - L + 1)
                test[s:s + L] = np.nan
                hidden.append((s, s + L))
            for method in errs:
                filled, _ = fill_gaps(test, half, method=method, max_gap=L)
                for s, e in hidden:
                    d = np.hypot(*(filled[s:e] - xy[s:e]).T)
                    if not np.isnan(d).any():
                        errs[method].append(d.max())
        out[L] = {m: (float(np.median(v)), float(np.percentile(v, 90)), len(v))
                  for m, v in errs.items() if v}
    return out


def team_repair_benchmark(z, n=150, seed=3, mode="swap"):
    """Inject realistic cross-team identity switches into clean stretches of the real tracking
    and check the repair finds them.

    mode "swap": at a real duel (a home and an away slot within 1.5 m on a frame where both
    teams have at most 11), the slot of team t follows the opponent from then on, the
    opponent's own slot ends, and the real player is re-acquired in a free slot of team t
    0-60 frames later -- the 12-vs-10 signature seen in the data.
    mode "birth": the tracker loses an opponent and re-acquires him 0-60 frames later in a
    new slot of the wrong team (team t).
    """
    from .players import split_identity_switches, smooth_and_differentiate
    from .teams import repair_team_labels
    rng = np.random.default_rng(seed)
    xy_all, valid_all = z["player_xy_raw"].astype(float), z["player_valid"]
    team_slot, half, pteam = z["team_of_slot"], z["half"], z["player_team"]
    # windows where the tracker's own labels are already plausible, so any repair inside the
    # window is a response to the injected switch
    c = np.stack([(valid_all & (team_slot == t)[None]).sum(1) for t in (0, 1)], 1)
    ok = (half > 0) & (c <= 11).all(1)
    okc = np.concatenate(([0], np.cumsum(ok)))
    res, tries = [], 0
    while len(res) < n and tries < 200000:
        tries += 1
        f = int(rng.integers(1600, len(ok) - 1600))
        lo, hi = f - 1500, f + 1500
        if okc[hi] - okc[lo] < hi - lo:
            continue
        X = np.where(valid_all[lo:hi, :, None], xy_all[lo:hi], np.nan)
        V = valid_all[lo:hi].copy()
        ff = f - lo
        t = int(rng.integers(0, 2))
        hs = np.flatnonzero((team_slot == t) & V[ff])
        os_ = np.flatnonzero((team_slot == 1 - t) & V[ff])
        if not len(hs) or not len(os_):
            continue
        d = np.hypot(*(X[ff][hs][:, None] - X[ff][os_][None]).transpose(2, 0, 1))
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > 1.5:
            continue
        h, o = hs[i], os_[j]
        end = ff + 1
        while end < len(V) and V[end, h] and V[end, o]:
            end += 1
        free = [s for s in np.flatnonzero(team_slot == t) if not V[:, s].any()]
        if end - ff < 100 or not free:
            continue
        delay, new = int(rng.integers(0, 60)), free[0]
        if mode == "swap":
            real_h = X[ff + 1:end, h].copy()
            X[ff + 1:end, h] = X[ff + 1:end, o]
            X[ff + 1:end, o] = np.nan
            V[ff + 1:end, o] = False
            X[ff + 1 + delay:end, new] = real_h[delay:]
            V[ff + 1 + delay:end, new] = True
            X[end:, h] = np.nan
            V[end:, h] = False
            wrong_slot = h
        else:
            real_o = X[ff + 1:end, o].copy()
            X[ff + 1:, o] = np.nan
            V[ff + 1:, o] = False
            X[ff + 1 + delay:end, new] = real_o[delay:]
            V[ff + 1 + delay:end, new] = True
            wrong_slot = new
        cc = np.stack([(V & (team_slot == tt)[None]).sum(1) for tt in (0, 1)], 1)
        detectable = int(np.sum((cc[:, t] > 11) & (cc[:, 1 - t] < 11))) >= C.TEAM_REPAIR_MIN_EPISODE
        rs = split_identity_switches(X)
        _, vel = smooth_and_differentiate(X, rs)
        seen = ~np.isnan(X[..., 0])
        js = rs.copy()
        js[0] = False
        js[1:] &= seen[:-1]
        tm, rep = repair_team_labels(X, V, team_slot, active=half[lo:hi] > 0, vel=vel,
                                     jump_start=js)
        frac = float(np.mean(tm[ff + 1 + delay:end, wrong_slot] == 1 - t))
        wrong = [r for r in rep if r["slot"] != wrong_slot and r["to"] > ff and r["from"] < end]
        hit = [r for r in rep if r["slot"] == wrong_slot]
        if mode == "birth" and hit:
            hit[0] = dict(hit[0], **{"from": hit[0]["from"] - delay})
        res.append((frac, len(wrong), (hit[0]["from"] - (ff + 1)) if hit else np.nan,
                    detectable))
    res = np.array(res, dtype=float)
    det = res[:, 3] > 0
    return {"n": int(len(res)), "detectable_frac": float(det.mean()),
            "recall": float(np.mean(res[det, 0] >= 0.9)),
            "wrong_relabel_rate": float(np.mean(res[:, 1] > 0)),
            "false_repair_when_undetectable": float(np.mean(res[~det, 1] > 0))
            if (~det).any() else None,
            "switch_point_abs_err_median": float(np.nanmedian(np.abs(res[:, 2])))}


def evaluate(path=DEFAULT_OUT, bench=True):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    graded = z["half"] > 0
    n_graded = int(graded.sum())
    state = z["ball_state"]
    pitch = z["pitch"]
    R = {"checks": []}

    def check(name, value, ok, target):
        R["checks"].append({"name": name, "value": value, "ok": bool(ok), "target": target})

    # 0. alignment -------------------------------------------------------------------------
    check("one tracking record per video frame (199,319)", int(z["atd_present"].sum()),
          bool(z["atd_present"].all()) and len(z["atd_present"]) == 199319, "all 199,319")

    # 1. coverage --------------------------------------------------------------------------
    cov = {k: float(np.mean(state[graded] == v)) for k, v in
           (("observed", OBSERVED), ("filled", FILLED), ("missing", MISSING))}
    cov["removed"] = float(np.mean(z["ball_removed"][graded]))
    raw_known = ~np.isnan(z["ball_xy_raw"][:, 0])
    known = ~np.isnan(z["ball_xy"][:, 0])
    cov["raw_known"] = float(raw_known[graded].mean())
    cov["clean_known"] = float(known[graded].mean())
    R["coverage"] = cov
    check("raw ball coverage matches DATA.md (59.3%)", round(cov["raw_known"] * 100, 1),
          abs(cov["raw_known"] - 0.593) < 0.005, "59.3 +/- 0.5 %")
    check("false detections removed", round(cov["removed"] * 100, 2),
          cov["removed"] < 0.03, "< 3 % of graded frames")

    # 2. gaps ------------------------------------------------------------------------------
    g_raw, g_clean = gap_lengths(raw_known, graded), gap_lengths(known, graded)
    buckets = [(1, 1), (2, 5), (6, 12), (13, 25), (26, 125), (126, 10**9)]
    R["gaps"] = {"{}-{}".format(lo, hi if hi < 10**9 else "inf"):
                 (int(((g_raw >= lo) & (g_raw <= hi)).sum()),
                  int(((g_clean >= lo) & (g_clean <= hi)).sum()))
                 for lo, hi in buckets}
    short_left = int(((g_clean >= 1) & (g_clean <= C.BALL_MAX_FILL_GAP)).sum())
    short_raw = int(((g_raw >= 1) & (g_raw <= C.BALL_MAX_FILL_GAP)).sum())
    R["short_gaps_unfilled"] = (short_left, short_raw)

    # 3. physics ---------------------------------------------------------------------------
    v_raw = implied_speed(np.where(graded[:, None], z["ball_xy_raw"], np.nan))
    v_clean = implied_speed(np.where(graded[:, None], z["ball_xy"], np.nan))
    R["ball_implied_speed"] = {
        "raw_p99": float(np.percentile(v_raw, 99)), "clean_p99": float(np.percentile(v_clean, 99)),
        "raw_frac_over_vmax": float(np.mean(v_raw > C.BALL_V_MAX)),
        "clean_frac_over_vmax": float(np.mean(v_clean > C.BALL_V_MAX))}
    check("ball steps faster than V_MAX after cleaning",
          round(R["ball_implied_speed"]["clean_frac_over_vmax"] * 100, 3),
          R["ball_implied_speed"]["clean_frac_over_vmax"] < 0.001, "< 0.1 % of steps")

    # 4. smoothing residual and speed agreement --------------------------------------------
    kept = graded & (state == OBSERVED)
    resid = np.hypot(*(z["ball_xy"][kept] - z["ball_xy_raw"][kept]).T)
    R["ball_smoothing_residual_m"] = {"median": float(np.median(resid)),
                                      "p95": float(np.percentile(resid, 95))}
    check("ball smoothing moves a sample (p95)", round(float(np.percentile(resid, 95)), 3),
          np.percentile(resid, 95) < 0.30, "< 0.30 m")

    # The ATD's ball speed column: documented as unusable rather than silently ignored.
    atd = z["ball_speed_atd"]
    raw_obs = graded & ~np.isnan(z["ball_xy_raw"][:, 0])
    present = raw_obs & ~np.isnan(atd)
    both = present & ~np.isnan(z["ball_speed"])
    R["atd_ball_speed_column"] = {
        "present_frac_of_ball_frames": float(present.sum() / max(1, raw_obs.sum())),
        "corr_with_motion": float(np.corrcoef(z["ball_speed"][both], atd[both])[0, 1])
        if both.sum() > 100 else None}

    # Where the ball disappears: a long gap that starts at the edge of the pitch is the ball
    # leaving play, which layer 3 (dead ball) leans on.
    gl = z["ball_gap_len"]
    starts = np.flatnonzero(graded[1:] & (gl[1:] > 125) & (gl[:-1] == 0)) + 1
    lx = z["ball_last_seen_xy"][starts]
    edge = ((lx[:, 0] < 3) | (lx[:, 0] > pitch[0] - 3) | (lx[:, 1] < 3) | (lx[:, 1] > pitch[1] - 3))
    R["long_gaps"] = {"n": int(len(starts)), "start_within_3m_of_line": float(edge.mean())
                      if len(starts) else None}

    # 5. players ---------------------------------------------------------------------------
    team = z["team_of_slot"]
    valid = z["player_valid"]
    pteam = z["player_team"]
    counts = np.stack([(pteam == t).sum(axis=1) for t in (0, 1)], axis=1)[graded]
    before = np.stack([valid[:, team == t].sum(axis=1) for t in (0, 1)], axis=1)[graded]
    R["players_per_team_before_repair"] = {
        "frac_over_11": float(np.mean(before > 11)),
        "frac_12_vs_10": float(np.mean(((before[:, 0] > 11) & (before[:, 1] < 11))
                                       | ((before[:, 1] > 11) & (before[:, 0] < 11))))}
    R["team_repairs"] = {"n": len(meta["repairs"]),
                         "frames": int(sum(r["to"] - r["from"] for r in meta["repairs"])),
                         "by_evidence": {k: int(sum(r["evidence"] == k for r in meta["repairs"]))
                                         for k in ("jump", "contact+motion", "start+reacquire", "contact",
                                                   "start")}}
    R["players_per_team"] = {
        "mean": counts.mean(axis=0).round(2).tolist(),
        "frac_9_to_11": float(np.mean((counts >= 9) & (counts <= 11))),
        "frac_over_11": float(np.mean(counts > 11)),
        "frac_under_8": float(np.mean(counts < 8)),
        "frac_12_vs_10": float(np.mean(((counts[:, 0] > 11) & (counts[:, 1] < 11))
                                       | ((counts[:, 1] > 11) & (counts[:, 0] < 11))))}
    check("12-vs-10 frames: before -> after repair (%)",
          "{} -> {}".format(round(R["players_per_team_before_repair"]["frac_12_vs_10"] * 100, 1),
                            round(R["players_per_team"]["frac_12_vs_10"] * 100, 1)),
          R["players_per_team"]["frac_12_vs_10"]
          < 0.35 * R["players_per_team_before_repair"]["frac_12_vs_10"], "cut by > 65 %")
    flagged = z["team_count_anomaly"][graded]
    check("every frame still over 11 is flagged team_count_anomaly",
          round(float(flagged.mean() * 100), 2),
          bool(np.all(flagged >= (counts > 11).any(axis=1))), "all flagged")

    R["player_duplicate_rate"] = float(z["player_duplicate"][graded].sum()
                                       / max(1, (~np.isnan(z["player_xy_raw"][graded][..., 0]))
                                             .sum()))
    pv = z["player_vel"][graded]
    pspeed = np.hypot(pv[..., 0], pv[..., 1])[valid[graded]]
    pspeed = pspeed[~np.isnan(pspeed)]
    R["player_speed"] = {"p99": float(np.percentile(pspeed, 99)),
                         "frac_over_11": float(np.mean(pspeed > 11))}
    check("player speeds over 11 m/s after cleaning", round(R["player_speed"]["frac_over_11"]
                                                           * 100, 3),
          R["player_speed"]["frac_over_11"] < 0.001, "< 0.1 %")
    patd = z["player_speed_atd"][graded][valid[graded]]
    pours = np.hypot(pv[..., 0], pv[..., 1])[valid[graded]]
    okp = ~np.isnan(patd) & ~np.isnan(pours)
    R["player_speed_vs_atd"] = {"corr": float(np.corrcoef(pours[okp], patd[okp])[0, 1]),
                                "median_abs_diff": float(np.median(np.abs(pours[okp] - patd[okp])))}
    check("player speed agrees with ATD speed column (corr)",
          round(R["player_speed_vs_atd"]["corr"], 3), R["player_speed_vs_atd"]["corr"] > 0.9,
          "> 0.90")

    # 6. attack direction ------------------------------------------------------------------
    R["direction"] = meta["direction"]
    dirs = meta["direction"]
    check("attack directions flip at half time and match the goalkeepers",
          {h: (d["home"], d["away"]) for h, d in dirs.items()},
          all(d["consistent"] for d in dirs.values())
          and dirs["1"]["home"] == -dirs["2"]["home"], "consistent + flipped")
    prov = meta.get("provider_direction_first_half", {})
    want = {"left_to_right": 1, "right_to_left": -1}
    if prov:
        agree = all(want.get(prov[k]) == dirs["1"][k] for k in ("home", "away") if k in prov)
        check("first-half directions agree with the provider metadata", prov, agree, "agree")

    # 7. ball out of the pitch --------------------------------------------------------------
    bx = z["ball_xy"][graded & known]
    outside = ((bx[:, 0] < -1) | (bx[:, 0] > pitch[0] + 1) | (bx[:, 1] < -1)
               | (bx[:, 1] > pitch[1] + 1))
    R["ball_outside_pitch_frac"] = float(outside.mean())

    if bench:
        for mode, label in (("swap", "mid-run identity switches"),
                            ("birth", "tracks born with the wrong team")):
            tb = team_repair_benchmark(z, mode=mode)
            R["team_repair_benchmark_" + mode] = tb
            check("team repair recovers injected {} (recall, of {} that break the count)"
                  .format(label, round(tb["detectable_frac"] * 100)) + "%",
                  round(tb["recall"], 3), tb["recall"] >= 0.9, ">= 0.90")
            # Within ~0.5 m a switch and a wrong-team re-acquisition leave identical data, so
            # some confusion between the two is irreducible from positions alone.
            check("team repair relabels a wrong slot ({})".format(mode),
                  round(tb["wrong_relabel_rate"], 3), tb["wrong_relabel_rate"] <= 0.10, "<= 0.10")

    # 8. gap-fill benchmark -----------------------------------------------------------------
    if bench:
        removed = z["ball_removed"]
        R["gap_fill_benchmark"] = gap_fill_benchmark(z["ball_xy_raw"].astype(np.float64),
                                                     z["half"], removed)
        b = R["gap_fill_benchmark"]
        L = C.BALL_MAX_FILL_GAP
        if L in b:
            p90 = b[L][meta["fill_method"]][1]
            check("gap fill p90 max-error at the longest filled gap ({} frames)".format(L),
                  round(p90, 2), p90 < 1.5, "< 1.5 m (control radius)")
    return R, z


def figures(R, z, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    graded = z["half"] > 0

    # trajectory sample: a busy 12 s stretch with gaps
    known_raw = ~np.isnan(z["ball_xy_raw"][:, 0])
    win = 300
    first = np.flatnonzero(graded)[0] + 25 * 60 * 10
    scores = [(np.sum(z["ball_state"][s:s + win] == FILLED), s)
              for s in range(first, first + 25 * 60 * 20, win)]
    s = max(scores)[1]
    t = np.arange(s, s + win)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for d, ax in enumerate(axes):
        ax.plot(t, z["ball_xy_raw"][t, d], ".", ms=4, color="0.6", label="raw")
        ax.plot(t, z["ball_xy"][t, d], "-", lw=1.2, color="C0", label="cleaned")
        f = z["ball_state"][t] == FILLED
        ax.plot(t[f], z["ball_xy"][t[f], d], "o", ms=3, color="C1", label="filled")
        r = z["ball_removed"][t]
        ax.plot(t[r], z["ball_xy_raw"][t[r], d], "x", ms=6, color="C3", label="removed")
        ax.set_ylabel(["ball x (m)", "ball y (m)"][d])
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("video frame")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ball_trajectory_sample.png"), dpi=110)
    plt.close(fig)

    if "gap_fill_benchmark" in R:
        b = R["gap_fill_benchmark"]
        Ls = sorted(b)
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, c in (("linear", "C0"), ("hermite", "C1")):
            ax.plot(Ls, [b[L][m][0] for L in Ls], "-", color=c, label=m + " median")
            ax.plot(Ls, [b[L][m][1] for L in Ls], "--", color=c, label=m + " p90")
        ax.axvline(C.BALL_MAX_FILL_GAP, color="0.5", lw=0.8)
        ax.axhline(1.5, color="C3", lw=0.8)
        ax.set_xlabel("hidden gap length (frames)")
        ax.set_ylabel("max position error in gap (m)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gap_fill_benchmark.png"), dpi=110)
        plt.close(fig)

    team = z["team_of_slot"]
    counts = np.stack([z["player_valid"][:, team == t].sum(axis=1) for t in (0, 1)], 1)[graded]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.arange(0, 16) - 0.5
    ax.hist(counts[:, 0], bins, alpha=0.6, label="home")
    ax.hist(counts[:, 1], bins, alpha=0.6, label="away")
    ax.set_xlabel("valid players per team on a graded frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "players_per_team.png"), dpi=110)
    plt.close(fig)


def write_report(R, out_dir):
    lines = ["# Layer 0 evaluation", "", "| check | value | target | |", "|---|---|---|---|"]
    for c in R["checks"]:
        lines.append("| {} | {} | {} | {} |".format(c["name"], c["value"], c["target"],
                                                  "PASS" if c["ok"] else "FAIL"))
    lines += ["", "```", json.dumps({k: v for k, v in R.items() if k != "checks"}, indent=1,
                                    default=str), "```"]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=DEFAULT_OUT)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()
    R, z = evaluate(args.npz, bench=not args.no_bench)
    figures(R, z, REPORT_DIR)
    write_report(R, REPORT_DIR)
    for c in R["checks"]:
        print("{:5}  {:<70} {}".format("PASS" if c["ok"] else "FAIL", c["name"], c["value"]))
    print("report:", os.path.relpath(os.path.join(REPORT_DIR, "report.md"), ROOT))


if __name__ == "__main__":
    main()
