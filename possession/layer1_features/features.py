"""Per-frame physics features, computed from the layer-0 arrays.

Everything is per video frame. Team-specific features come in home/away pairs ("_h", "_a");
positions along the pitch are measured in each team's attacking frame (0 = own goal line,
pitch length = the goal it attacks), so the two halves need no special casing.

Ball features are NaN where the ball is unknown (41% of graded frames). Team-shape features
need no ball and are always available - they carry the signal through the gaps.
"""

import json

import numpy as np

from .. import config as C
from ..common import distance_outside_pitch, distance_to_line, restart_spots


def _nearest(dist, team, t):
    """Distance and slot of team t's player nearest to the ball, per frame."""
    d = np.where(team == t, dist, np.inf)
    slot = np.argmin(d, axis=1)
    best = d[np.arange(len(d)), slot]
    best = np.where(np.isfinite(best), best, np.nan)
    return best, np.where(np.isnan(best), -1, slot)


def _dwell_fast(mask):
    """Frames a condition has held continuously, ending at each frame (0 where it is False)."""
    idx = np.arange(len(mask))
    last_false = np.where(~mask, idx, -1)
    last_false = np.maximum.accumulate(last_false)
    return np.where(mask, idx - last_false, 0).astype(np.int32)


def rolling_mean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN; NaN where the window is empty."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.clip(i - half_width, 0, len(x))
    hi = np.clip(i + half_width + 1, 0, len(x))
    n = cn[hi] - cn[lo]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def carry(x, backward=False):
    """Last (or next) known value of x, carried across NaN."""
    x = np.asarray(x, float)
    if backward:
        return carry(x[::-1])[::-1]
    idx = np.where(~np.isnan(x), np.arange(len(x)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, x[np.clip(idx, 0, None)], np.nan)


def team_shape(xy, team, attack_sign, half, length):
    """Centroid, spread, defensive line and forward speed of each team, attacking frame."""
    out = {}
    sign = np.zeros((len(half), 2))
    for t in (0, 1):
        sign[:, t] = attack_sign[half, t]
    for t, tag in ((0, "h"), (1, "a")):
        mine = team == t
        x = np.where(mine, xy[..., 0], np.nan)
        y = np.where(mine, xy[..., 1], np.nan)
        s = sign[:, t][:, None]
        # attacking-frame x: distance from own goal line
        xa = np.where(s > 0, x, length - x)
        with np.errstate(all="ignore"):
            cx = np.nanmean(xa, axis=1)
            out["centroid_x_" + tag] = cx
            out["spread_x_" + tag] = np.nanstd(xa, axis=1)
            out["spread_y_" + tag] = np.nanstd(y, axis=1)
            srt = np.sort(xa, axis=1)  # NaN last
            # deepest outfield player = second deepest overall (the deepest is the keeper)
            out["def_line_" + tag] = srt[:, 1]
            n = np.sum(mine, axis=1)
            out["front_line_" + tag] = srt[np.arange(len(srt)), np.clip(n - 1, 0, None)]
        out["n_players_" + tag] = n.astype(np.int8)
    return out, sign


def ball_player(ball_xy, ball_vel, xy, vel, team, sign, length):
    """Nearest player of each team to the ball: distance, relative speed, closing speed,
    and how many players of each team are close."""
    out = {}
    diff = xy - ball_xy[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(team >= 0, dist, np.nan)
    for t, tag in ((0, "h"), (1, "a")):
        d, slot = _nearest(np.nan_to_num(dist, nan=np.inf), team, t)
        out["d_" + tag] = d
        out["slot_" + tag] = slot.astype(np.int8)
        rows = np.arange(len(d))
        pv = vel[rows, np.clip(slot, 0, None)]
        pv = np.where((slot >= 0)[:, None], pv, np.nan)
        out["relv_" + tag] = np.hypot(*(ball_vel - pv).T)
        # closing speed: how fast the gap to the ball is shrinking (m/s, + = approaching)
        rel = ball_xy - xy[rows, np.clip(slot, 0, None)]
        rel = np.where((slot >= 0)[:, None], rel, np.nan)
        with np.errstate(all="ignore"):
            unit = rel / np.hypot(*rel.T)[:, None]
        out["closing_" + tag] = np.sum((pv - ball_vel) * unit, axis=1)
        # player speed towards his attacking goal (m/s)
        out["nearest_fwd_speed_" + tag] = pv[:, 0] * sign[:, t]
        for r in C.PRESSURE_RADII:
            close = (team == t) & (dist < r)
            out["n{}_{}".format(int(r), tag)] = close.sum(axis=1).astype(np.int8)
    for t, tag in ((0, "h"), (1, "a")):
        within = out["d_" + tag] < C.CONTROL_RADIUS
        out["dwell_" + tag] = _dwell_fast(np.nan_to_num(within, nan=0).astype(bool))
    out["margin"] = out["d_a"] - out["d_h"]
    return out


def dead_ball_cues(z, ball_xy, length, width, sign):
    out = {}
    known = ~np.isnan(ball_xy[:, 0])
    out["ball_known"] = known
    out["ball_speed"] = z["ball_speed"].astype(float)
    out["ball_outside"] = distance_outside_pitch(ball_xy, length, width)
    out["ball_to_line"] = distance_to_line(ball_xy, length, width)
    spots = np.array(list(restart_spots(length, width).values()))
    d = np.hypot(*(ball_xy[:, None, :] - spots[None]).transpose(2, 0, 1))
    with np.errstate(all="ignore"):
        out["ball_to_restart_spot"] = np.nanmin(d, axis=1) if known.any() else np.nan
    out["ball_to_restart_spot"] = np.where(known, out["ball_to_restart_spot"], np.nan)
    still = known & (out["ball_speed"] < C.BALL_STILL_SPEED)
    out["ball_still_frames"] = _dwell_fast(still)
    out["ball_frames_since_seen"] = z["ball_frames_since_seen"]
    out["ball_frames_until_seen"] = z["ball_frames_until_seen"]
    out["ball_gap_len"] = z["ball_gap_len"]
    lx = z["ball_last_seen_xy"].astype(float)
    out["last_seen_to_line"] = distance_to_line(lx, length, width)
    out["last_seen_outside"] = distance_outside_pitch(lx, length, width)
    # ball position along the pitch in home's attacking frame (0 = home goal line)
    out["ball_x_home_frame"] = np.where(sign[:, 0] > 0, ball_xy[:, 0], length - ball_xy[:, 0])
    return out


def pressing(xy, vel, team, sign, length):
    """Players-only evidence: how tightly each team marks, how far it has pushed, whether it is
    running forward. The team without the ball presses and marks; the team with it spreads and
    moves up."""
    out = {}
    for t, tag in ((0, "h"), (1, "a")):
        mine, theirs = team == t, team == 1 - t
        d = np.hypot(xy[:, :, None, 0] - xy[:, None, :, 0], xy[:, :, None, 1] - xy[:, None, :, 1])
        d = np.where(mine[:, :, None] & theirs[:, None, :], d, np.inf)
        nearest = d.min(axis=2)
        nearest = np.where(mine & np.isfinite(nearest), nearest, np.nan)
        with np.errstate(all="ignore"):
            out["press_" + tag] = np.nanmean(nearest, axis=1)
            s = sign[:, t][:, None]
            xa = np.where(s > 0, xy[..., 0], length - xy[..., 0])
            out["n_opp_half_" + tag] = np.sum(mine & (xa > length / 2), axis=1).astype(np.int8)
            fwd = vel[..., 0] * s
            n = np.sum(mine & ~np.isnan(fwd), axis=1)
            out["fwd_share_" + tag] = np.where(n > 0, np.sum(mine & (fwd > 1.0), axis=1) / np.maximum(n, 1),
                                                np.nan)
    return out


def movement(vel, team):
    """How fast everybody is moving - play stops at a dead ball."""
    with np.errstate(all="ignore"):
        sp = np.hypot(vel[..., 0], vel[..., 1])
        sp = np.where(team >= 0, sp, np.nan)
        return {"mean_speed_all": np.nanmean(sp, axis=1),
                "mean_speed_h": np.nanmean(np.where(team == 0, sp, np.nan), axis=1),
                "mean_speed_a": np.nanmean(np.where(team == 1, sp, np.nan), axis=1)}


def build_features(z, st):
    meta = json.loads(str(z["meta"]))
    length, width = [float(v) for v in z["pitch"]]
    half = z["half"].astype(int)
    xy = z["player_xy"].astype(float)
    vel = z["player_vel"].astype(float)
    team = z["player_team"]
    ball_xy = z["ball_xy"].astype(float)
    ball_vel = z["ball_vel"].astype(float)

    F = {}
    shape, sign = team_shape(xy, team, z["attack_sign"], half, length)
    F.update(shape)
    # forward speed of each team's centroid (m/s, + = towards the goal it attacks)
    for t, tag in ((0, "h"), (1, "a")):
        with np.errstate(all="ignore"):
            vx = np.nanmean(np.where(team == t, vel[..., 0], np.nan), axis=1)
        F["centroid_fwd_speed_" + tag] = vx * sign[:, t]
    F["territory"] = F["centroid_x_h"] - F["centroid_x_a"]   # + = home pushed up
    F["line_gap"] = F["def_line_h"] - F["def_line_a"]
    F.update(ball_player(ball_xy, ball_vel, xy, vel, team, sign, length))
    F.update(dead_ball_cues(z, ball_xy, length, width, sign))
    # where the ball is relative to each team's block (+ = ball ahead of the team's centroid)
    bx_h = F["ball_x_home_frame"]
    F["ball_ahead_h"] = bx_h - F["centroid_x_h"]
    F["ball_ahead_a"] = (length - bx_h) - F["centroid_x_a"]
    F["area_h"] = F["spread_x_h"] * F["spread_y_h"]
    F["area_a"] = F["spread_x_a"] * F["spread_y_a"]
    # ball travelling towards the goal home attacks (m/s): after a turnover the ball heads the
    # other way almost at once
    F["ball_fwd_speed_home"] = ball_vel[:, 0] * sign[:, 0]
    F["ball_fwd_1s"] = rolling_mean(F["ball_fwd_speed_home"], 12)
    # multi-scale ball evidence: who has been nearest, over 1 s and 3 s windows
    near_home = np.where(np.isnan(F["margin"]), np.nan, (F["margin"] > 0).astype(float))
    for w, tag in ((12, "1s"), (37, "3s"), (125, "10s")):
        F["margin_" + tag] = rolling_mean(np.clip(F["margin"], -10, 10), w)
        F["near_share_" + tag] = rolling_mean(near_home, w)
    # through ball gaps: the evidence on either side of the gap
    F["margin_before_gap"] = carry(np.clip(F["margin_1s"], -10, 10))
    F["margin_after_gap"] = carry(np.clip(F["margin_1s"], -10, 10), backward=True)
    F.update(movement(vel, team))
    # pressing needs an n x 46 x 46 distance array: do it in chunks
    chunks = []
    for a in range(0, len(half), 20000):
        b = min(len(half), a + 20000)
        chunks.append(pressing(xy[a:b], vel[a:b], team[a:b], sign[a:b], length))
    for k in chunks[0]:
        F[k] = np.concatenate([c[k] for c in chunks])
    F["length_h"] = F["front_line_h"] - F["def_line_h"]
    F["length_a"] = F["front_line_a"] - F["def_line_a"]
    F.update(st)
    F["team_count_anomaly"] = z["team_count_anomaly"]
    F["half"] = z["half"]
    return F, meta


# The set later layers should use, chosen with layer1_features/evaluate.py: every feature here
# adds evidence, and none is a near-duplicate (|r| > 0.95) of another. Home-minus-away
# differences are built from the "_h"/"_a" pairs by the consumer.
RECOMMENDED = {
    # fast: line up with turnovers within ~0-2 s
    "ball": ["margin", "d_h", "d_a", "relv_h", "relv_a", "closing_h", "closing_a",
             "n2_h", "n2_a", "n5_h", "n5_a", "dwell_h", "dwell_a",
             "nearest_fwd_speed_h", "nearest_fwd_speed_a", "ball_x_home_frame",
             "ball_ahead_h", "ball_ahead_a", "ball_fwd_speed_home", "ball_fwd_1s",
             "margin_1s", "margin_3s", "margin_10s", "near_share_1s", "near_share_3s",
             "near_share_10s", "margin_before_gap", "margin_after_gap"],
    # slow (lag turnovers by 6-10 s) but always available, including through ball gaps
    "shape": ["territory", "spread_x_h", "spread_x_a", "spread_y_h", "spread_y_a",
              "area_h", "area_a", "centroid_fwd_speed_h", "centroid_fwd_speed_a",
              "press_h", "press_a", "n_opp_half_h", "n_opp_half_a", "fwd_share_h", "fwd_share_a",
              "length_h", "length_a"],
    # for the dead-ball layer
    "dead_ball": ["ball_known", "ball_speed", "ball_outside", "ball_to_line",
                  "ball_to_restart_spot", "ball_still_frames", "ball_frames_since_seen",
                  "ball_frames_until_seen", "ball_gap_len", "last_seen_to_line",
                  "last_seen_outside", "mean_speed_all", "mean_speed_h", "mean_speed_a"],
    "smart_tagging": ["st_possessor", "st_set_piece", "st_set_piece_team"],
}
