"""Read the raw ATD (FIFA/EPTS) into numpy arrays, indexed by video frame.

Raw line format:  <frame_id>:<slot>;<slot>;...;<slot>:<ball>
each slot / ball group is comma-separated numbers ("x,y,speed"), NaN where unseen.
Positions are normalised 0-1; frame_id counts from 1, video frames from 0.

One pass, vectorised: separators are rewritten to commas and the whole file goes through
pandas' C parser, which is ~20x faster than splitting line by line in Python.
"""

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1


def read_meta(meta_path):
    """(team_of_slot array, (home_name, away_name)). First team in the file is home."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    names = tuple(t.findtext("Name") for t in teams)
    return team_of_slot, names


def read_meta_directions(meta_path):
    """The provider's own first-half attack direction per team, e.g. {"home": "left_to_right"}."""
    root = ET.parse(meta_path).getroot()
    out = {}
    for side, t in zip(("home", "away"), root.iter("Team")):
        for p in t.iter("ProviderParameter"):
            if p.findtext("Name") == "attack_direction_first_half":
                out[side] = p.findtext("Value")
    return out


def _group_widths(line):
    """Number of fields in a player group and in the ball group, read off one line."""
    _, players, ball = line.rstrip("\n").split(":")
    slot_width = len(players.split(";")[0].split(","))
    ball_width = len(ball.split(","))
    n_slots = len(players.split(";"))
    return n_slots, slot_width, ball_width


def read_raw(raw_path, n_frames):
    """Parse the raw ATD.

    Returns a dict of arrays indexed by video frame (length n_frames):
      players  (n_frames, n_slots, slot_width)  normalised x, y, speed[, ...]
      ball     (n_frames, ball_width)           normalised x, y, speed[, ...]
      present  (n_frames,) bool                 a record exists for this frame
    """
    with open(raw_path) as handle:
        text = handle.read()
    first = text[:text.index("\n")]
    n_slots, slot_width, ball_width = _group_widths(first)

    table = pd.read_csv(io.StringIO(text.replace(":", ",").replace(";", ",")),
                        header=None, dtype=np.float64, engine="c")
    expected_cols = 1 + n_slots * slot_width + ball_width
    if table.shape[1] != expected_cols:
        raise ValueError("ATD has {} columns per line, expected {} ({} slots x {} + ball {})"
                         .format(table.shape[1], expected_cols, n_slots, slot_width, ball_width))

    values = table.to_numpy()
    frame = values[:, 0].astype(np.int64) - 1  # frame_id counts from 1
    if frame.min() < 0 or frame.max() >= n_frames:
        raise ValueError("frame ids out of range: {}..{}".format(frame.min() + 1, frame.max() + 1))

    players = np.full((n_frames, n_slots, slot_width), np.nan, dtype=np.float64)
    ball = np.full((n_frames, ball_width), np.nan, dtype=np.float64)
    present = np.zeros(n_frames, dtype=bool)
    players[frame] = values[:, 1:1 + n_slots * slot_width].reshape(-1, n_slots, slot_width)
    ball[frame] = values[:, 1 + n_slots * slot_width:]
    present[frame] = True
    return {"players": players, "ball": ball, "present": present,
            "n_slots": n_slots, "slot_width": slot_width, "ball_width": ball_width}
