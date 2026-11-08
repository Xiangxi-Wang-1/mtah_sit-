"""Layer 0 - clean the data.

    python -m possession.layer0_clean.run            # writes cache/layer0_tactical.npz

Input : the raw tactical ATD (data/tactical/BAR-ATH_tactical_atd.{xml,txt})
Output: one .npz with every array indexed by video frame (= submission row), in metres.

Ball
  ball_xy, ball_vel, ball_speed     cleaned ball (NaN where unknown)
  ball_xy_raw                       the ball as delivered, in metres
  ball_state                        0 missing, 1 filled, 2 observed, 3 removed as false
  ball_removed                      delivered detection dropped (over-speed or isolated spike)
  ball_frames_since_seen            frames since the cleaned ball was last known (0 if known)
  ball_frames_until_seen            frames until it is next known (0 if known)
  ball_gap_len                      length of the gap this frame sits in (0 if known)
  ball_last_seen_xy, ball_next_seen_xy   where the ball was before / after the gap
Players
  player_xy, player_vel             (n_frames, n_slots, 2) cleaned positions, velocities
  player_valid                      present, not a duplicate
  player_team                       (n_frames, n_slots) 0 home, 1 away, -1 not valid; after
                                    the cross-team identity-switch repair
  team_count_anomaly                a team still shows 12+ after repair: team labels on this
                                    frame are not fully trustworthy (no switch point found)
  player_xy_raw, player_duplicate, player_speed_atd, team_of_slot (the tracker's own team)
Match
  half                              1, 2 on graded frames, 0 elsewhere
  attack_sign                       (3, 2): attack_sign[half, team] = +1 if attacking +x
  pitch, meta (json: teams, directions, repairs)

Not used: the ATD's ball speed column. It is present on only 7% of ball frames and does not
track the ball's motion (see reports/layer0/report.md).

The cache is derived frame-level tracking: it is gitignored and must never be committed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

from .parse import read_meta, read_meta_directions, read_raw  # noqa: E402
from .ball import clean_ball  # noqa: E402
from .players import clean_players  # noqa: E402
from .teams import repair_team_labels  # noqa: E402
from .direction import attack_direction  # noqa: E402
from .. import config as C  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "cache", "layer0_tactical.npz")


def half_array(n_frames, game=GAME):
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return half


def build(game=GAME, data_dir=None, fill_method=None, repair=True, verbose=True):
    fill_method = fill_method or C.BALL_FILL_METHOD
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width])

    team_of_slot, names = read_meta(meta_path)
    provider_dirs = read_meta_directions(meta_path)
    raw = read_raw(raw_path, n_frames)
    if raw["n_slots"] != len(team_of_slot):
        raise ValueError("{} slots in raw ATD, {} in metadata".format(
            raw["n_slots"], len(team_of_slot)))
    if verbose:
        print("parsed {} frames, {} slots  ({:.0f}s)".format(
            int(raw["present"].sum()), raw["n_slots"], time.time() - t0))
    return clean(raw, team_of_slot, names, provider_dirs, half_array(n_frames, game),
                 (length, width), game, fill_method, repair, verbose, t0)


def clean(raw, team_of_slot, names, provider_dirs, half, pitch, game, fill_method=None,
          repair=True, verbose=True, t0=None):
    """Clean one game from normalised raw arrays (any source: the ATD, or a bench game).

    raw: {"players": (n, slots, >=2) normalised x, y[, speed], "ball": (n, >=2), "present": (n,)}
    half: (n,) 1 / 2 on graded frames, 0 elsewhere.  pitch: (length, width) in metres.
    """
    fill_method = fill_method or C.BALL_FILL_METHOD
    t0 = t0 or time.time()
    length, width = pitch
    scale = np.array([length, width])
    n_frames = len(half)
    def col(a, k):
        return a[..., k] if a.shape[-1] > k else np.full(a.shape[:-1], np.nan)
    ball_raw = raw["ball"][:, :2] * scale
    ball = clean_ball(ball_raw, half, fill_method=fill_method)
    players_raw = raw["players"][..., :2] * scale
    players = clean_players(players_raw, team_of_slot)

    if repair:
        team, repairs = repair_team_labels(players["xy"], players["valid"], team_of_slot,
                                           active=half > 0, vel=players["vel"],
                                           jump_start=players["jump_start"], verbose=verbose)
    else:
        team = np.where(players["valid"], team_of_slot[None, :], -1).astype(np.int8)
        repairs = []

    counts = np.stack([(team == t).sum(axis=1) for t in (0, 1)], axis=1)
    team_count_anomaly = (counts > 11).any(axis=1)

    direction = attack_direction(players["xy"], players["valid"], team_of_slot, half, length)
    attack_sign = np.zeros((3, 2), dtype=np.int8)
    for h, d in direction.items():
        attack_sign[h] = (d["home"], d["away"])

    if verbose:
        print("cleaned, {} team repairs  ({:.0f}s)".format(len(repairs), time.time() - t0))
    f32 = np.float32
    return {
        "ball_xy": ball["xy"].astype(f32),
        "ball_vel": ball["vel"].astype(f32),
        "ball_speed": ball["speed"].astype(f32),
        "ball_state": ball["state"],
        "ball_removed": ball["removed"],
        "ball_xy_raw": ball_raw.astype(f32),
        "ball_frames_since_seen": ball["frames_since_seen"],
        "ball_frames_until_seen": ball["frames_until_seen"],
        "ball_gap_len": ball["gap_len"],
        "ball_last_seen_xy": ball["last_seen_xy"].astype(f32),
        "ball_next_seen_xy": ball["next_seen_xy"].astype(f32),
        "ball_speed_atd": col(raw["ball"], 2).astype(f32),
        "player_xy": players["xy"].astype(f32),
        "player_xy_raw": players_raw.astype(f32),
        "player_vel": players["vel"].astype(f32),
        "player_valid": players["valid"],
        "player_team": team,
        "team_count_anomaly": team_count_anomaly,
        "player_duplicate": players["duplicate"],
        "player_speed_atd": col(raw["players"], 2).astype(f32),
        "team_of_slot": team_of_slot,
        "atd_present": raw["present"],
        "half": half,
        "attack_sign": attack_sign,
        "pitch": np.array([length, width]),
        "meta": np.array(json.dumps({"game": game, "teams": names, "direction": direction,
                                     "provider_direction_first_half": provider_dirs,
                                     "fill_method": fill_method, "repairs": repairs})),
    }


def main():
    parser = argparse.ArgumentParser(description="Layer 0: clean the tactical ATD")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--fill", default=None, choices=["linear", "hermite"])
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    result = build(data_dir=args.data_dir, fill_method=args.fill, repair=not args.no_repair)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **result)
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
