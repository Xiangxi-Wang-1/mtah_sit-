"""Read Metrica Sports' three open sample games into plain arrays.

    https://github.com/metrica-sports/sample-data   (clone it; pass its path, or set BENCH_SRC)

Games 1 and 2 are CSV (one tracking file per team, ball in both, plus RawEventsData.csv); game 3 is
FIFA/EPTS (tracking.txt + metadata.xml) with events in JSON. All three are 25 fps with positions
normalised 0-1 on a 105 x 68 pitch. These are other games, which RULES.md allows for training and
validation; nothing here touches the challenge match.

load(game) -> {
  players  (n, slots, 2) normalised x, y; NaN where the player is not on the pitch
  ball     (n, 2)
  team_of_slot (slots,) 0 home, 1 away
  period   (n,) 1, 2 (0 outside a period)
  events   list of {team 0/1/None, type, subtype, period, start, end} with frames 0-based
  names    (home, away), pitch (105, 68)
}
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

SRC = os.environ.get("BENCH_SRC", "/home/claude/bench_src/sample-data")
PITCH = (105.0, 68.0)


def _dir(game):
    return os.path.join(SRC, "data", "Sample_Game_{}".format(game))


def _csv_team(path):
    with open(path) as f:
        f.readline(); f.readline()
        header = f.readline().rstrip("\n").split(",")
    t = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
    cols = [i for i, h in enumerate(header) if h.startswith("Player")]
    xy = np.stack([t[:, [c, c + 1]] for c in cols], axis=1)
    b = header.index("Ball")
    return t[:, 0].astype(int), t[:, 1].astype(int), xy, t[:, [b, b + 1]]


def _load_csv(game):
    d = _dir(game)
    stem = os.path.join(d, "Sample_Game_{}_".format(game))
    per, frame, home, ball = _csv_team(stem + "RawTrackingData_Home_Team.csv")
    per2, frame2, away, _ = _csv_team(stem + "RawTrackingData_Away_Team.csv")
    assert np.array_equal(frame, frame2)
    n = frame.max()
    players = np.full((n, home.shape[1] + away.shape[1], 2), np.nan)
    players[frame - 1] = np.concatenate([home, away], axis=1)
    B = np.full((n, 2), np.nan)
    B[frame - 1] = ball
    period = np.zeros(n, np.int8)
    period[frame - 1] = per
    team_of_slot = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    ev = pd.read_csv(stem + "RawEventsData.csv")
    events = [{"team": {"Home": 0, "Away": 1}.get(r["Team"]), "type": r["Type"],
               "subtype": "" if pd.isna(r["Subtype"]) else str(r["Subtype"]),
               "period": int(r["Period"]), "start": int(r["Start Frame"]) - 1,
               "end": int(r["End Frame"]) - 1 if r["End Frame"] > 0 else int(r["Start Frame"]) - 1}
              for _, r in ev.iterrows()]
    return {"players": players, "ball": B, "team_of_slot": team_of_slot, "period": period,
            "events": events, "names": ("Home", "Away"), "pitch": PITCH}


def _load_epts(game):
    d = _dir(game)
    stem = os.path.join(d, "Sample_Game_{}_".format(game))
    root = ET.parse(stem + "metadata.xml").getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    team_of = {p.get("id"): teams.index(p.get("teamId")) for p in root.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    order = [c.get("playerId") for c in root.iter("PlayerChannel") if c.get("channelId") == "x"]
    col = {pid: i for i, pid in enumerate(order)}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    with open(stem + "tracking.txt") as f:
        text = f.read()
    t = pd.read_csv(pd.io.common.StringIO(text.replace(":", ",").replace(";", ",")),
                    header=None, dtype=float).to_numpy()
    frame = t[:, 0].astype(int)
    n = frame.max()
    S = len(order)
    players = np.full((n, S, 2), np.nan)
    B = np.full((n, 2), np.nan)
    # the columns change at each substitution: one DataFormatSpecification per frame range
    for spec in root.iter("DataFormatSpecification"):
        a, b = int(spec.get("startFrame")), int(spec.get("endFrame"))
        refs = [chan[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        rows = (frame >= a) & (frame <= b)
        k = len(refs)
        vals = t[rows, 1:1 + 2 * k].reshape(-1, k, 2)
        idx = frame[rows] - 1
        for j, pid in enumerate(refs):
            players[idx, col[pid]] = vals[:, j]
        B[idx] = t[rows, 1 + 2 * k:3 + 2 * k]
    period = np.zeros(n, np.int8)
    for h, key in ((1, "first_half"), (2, "second_half")):
        a, b = int(params[key + "_start"]), int(params[key + "_end"])
        period[a - 1:b] = h
    names = tuple(t.findtext("Name") for t in root.iter("Team"))
    events = []
    for e in json.load(open(stem + "events.json"))["data"]:
        st = e["subtypes"]
        if isinstance(st, dict):
            st = st["name"]
        elif isinstance(st, list):
            st = "-".join(s["name"] for s in st)
        else:
            st = ""
        team = e["team"]["name"] if e.get("team") else None
        events.append({"team": names.index(team) if team in names else None,
                       "type": e["type"]["name"], "subtype": st, "period": int(e["period"]),
                       "start": int(e["start"]["frame"]) - 1,
                       "end": int(e["end"]["frame"] or e["start"]["frame"]) - 1})
    team_of_slot = np.array([team_of[p] for p in order], np.int8)
    return {"players": players, "ball": B, "team_of_slot": team_of_slot, "period": period,
            "events": events, "names": names, "pitch": PITCH}


def load(game):
    return _load_epts(game) if game == 3 else _load_csv(game)
