"""Small helpers shared by every layer: cache paths, Smart Tagging, pitch geometry."""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
REPORTS = os.path.join(ROOT, "reports")


# Which game the layers read and write. "tactical" is the challenge match; a bench game (see
# possession/bench) sets POSSESSION_GAME=bench1 etc., so the same layer code runs on it unchanged.
TAG = os.environ.get("POSSESSION_GAME", "tactical")


def cache_path(layer, source=None):
    return os.path.join(CACHE, "{}_{}.npz".format(layer, source or TAG))


def load_layer(layer, source=None):
    path = cache_path(layer, source)
    if not os.path.isfile(path):
        raise IOError("{} not found - run that layer first".format(os.path.relpath(path, ROOT)))
    return np.load(path, allow_pickle=False)


# --- Smart Tagging ------------------------------------------------------------------------------
# Which side a phase code says is in possession, relative to the phase's Team tag. Read off the
# data, not assumed: on "attacking" codes the tagged team's player is nearest the ball 63-76% of
# the time; on "defending" codes only 24-37% (so the opponent has it). The codes paired with
# them (DEFENDING IN MIDDLE THIRD vs PROGRESSION, 66 each, ...) are the same moments seen from
# the other side. Location-only codes (BALL IN FINAL THIRD, PLAYERS IN THE BOX, ...) say nothing
# about the side and are left out.
POSSESSOR = {
    "BUILD UP": +1, "PROGRESSION": +1, "CREATING CHANCES": +1, "ATTACKING TRANSITION": +1,
    "DEFENDING IN DEFENSIVE THIRD": -1, "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENDING IN ATTACKING THIRD": -1, "DEFENSIVE TRANSITION": -1,
}
SET_PIECE_TYPES = ["Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)",
                   "Kick Off (Start)"]


def load_smart_tagging(game=GAME, data_dir=None):
    """[{code, start, end (video frames, end exclusive), team (0 home, 1 away, None), type}]."""
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    root = ET.parse(path).getroot()
    teams = {}
    out = []
    for inst in root.iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({"code": (inst.findtext("code") or "").strip().upper(),
                    "start": int(round(float(inst.findtext("start")) * fps)),
                    "end": int(round(float(inst.findtext("end")) * fps)),
                    "team_name": tags.get("Team"), "type": tags.get("Type")})
        teams.setdefault(tags.get("Team"), 0)
    return out


def team_index_from_names(home_name, away_name):
    """Map Smart Tagging team names to 0/1. The tagging spells Athletic 'Atletic Club', so match
    on the first word that is not a generic club word."""
    def key(name):
        words = [w for w in (name or "").lower().replace(".", "").split()
                 if w not in ("fc", "club", "cf", "de", "sd", "ud")]
        return words[0][:5] if words else ""
    return {key(home_name): 0, key(away_name): 1}, key


def smart_tagging_arrays(n_frames, home_name, away_name, game=GAME, data_dir=None):
    """Per-frame arrays from the Smart Tagging.

    st_possessor  +1 a phase says home has the ball, -1 away, 0 no phase (or both - rare)
    st_set_piece  0 none, else 1 + index into SET_PIECE_TYPES of the set piece window here
    st_set_piece_team  0 home, 1 away, -1 unknown (every throw-in is tagged N/A)
    """
    if TAG != "tactical":       # a bench game: tags were simulated when the bench was built
        st = np.load(os.path.join(CACHE, "bench", TAG + "_smart_tagging.npz"))
        return {k: st[k] for k in ("st_possessor", "st_set_piece", "st_set_piece_team")}
    mapping, key = team_index_from_names(home_name, away_name)
    home_votes = np.zeros(n_frames, np.int16)
    away_votes = np.zeros(n_frames, np.int16)
    sp = np.zeros(n_frames, np.int8)
    sp_team = np.full(n_frames, -1, np.int8)
    for e in load_smart_tagging(game, data_dir):
        a, b = max(0, e["start"]), min(n_frames, e["end"])
        t = mapping.get(key(e["team_name"]))
        if e["code"] in POSSESSOR and t is not None:
            side = t if POSSESSOR[e["code"]] > 0 else 1 - t
            (home_votes if side == 0 else away_votes)[a:b] += 1
        if e["code"] == "SET PIECES" and e["type"] in SET_PIECE_TYPES:
            sp[a:b] = 1 + SET_PIECE_TYPES.index(e["type"])
            if t is not None:
                sp_team[a:b] = t
    poss = np.sign(home_votes.astype(int) - away_votes.astype(int)).astype(np.int8)
    return {"st_possessor": poss, "st_set_piece": sp, "st_set_piece_team": sp_team}


# --- pitch geometry -----------------------------------------------------------------------------
def restart_spots(length, width):
    """Named (x, y) points where play restarts."""
    return {
        "centre": (length / 2, width / 2),
        "corner_00": (0.0, 0.0), "corner_0w": (0.0, width),
        "corner_l0": (length, 0.0), "corner_lw": (length, width),
        "penalty_0": (11.0, width / 2), "penalty_l": (length - 11.0, width / 2),
    }


def distance_outside_pitch(xy, length, width):
    """How far a point is beyond the touch/goal lines (0 inside)."""
    dx = np.maximum(0, np.maximum(-xy[..., 0], xy[..., 0] - length))
    dy = np.maximum(0, np.maximum(-xy[..., 1], xy[..., 1] - width))
    return np.hypot(dx, dy)


def distance_to_line(xy, length, width):
    """Distance to the nearest touch or goal line (inside the pitch), 0 outside."""
    d = np.minimum.reduce([xy[..., 0], length - xy[..., 0], xy[..., 1], width - xy[..., 1]])
    return np.maximum(0, d)
