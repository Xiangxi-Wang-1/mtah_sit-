"""Frame-level H / A / D answer keys for the bench games, from Metrica's own event labelling.

Every rule below is taken from Metrica's event definitions (sample-data/documentation/
events-definitions.pdf) and from CHALLENGE.md; nothing is fitted to anything.

Possession (Metrica: "a player is regarded as being in possession as soon as they touch the ball
... until another player takes over possession, a shot is deflected off the woodwork or another
player, the ball is significantly deflected by the referee, or the ball goes out of play")
  plays the ball    PASS, CARRY, SHOT, SET PIECE, BALL LOST, RECOVERY by a team, at the event's
                    start frame, and a completed PASS at its end frame (the receiver's touch)
  no change         CHALLENGE (won or lost: the touch that wins it is tagged as its own
                    RECOVERY), FAULT RECEIVED, CARD, BALL OUT
  team at frame t   the team of the most recent "plays the ball" moment <= t

Dead ball (Metrica: "out of play: the ball went out of bounds or the referee has stopped play")
  stops play        BALL OUT (at its end: the ball crossing the line); SHOT ending OUT or GOAL
                    (at its end); FAULT RECEIVED and a CHALLENGE with FAULT but not ADVANTAGE (at
                    the foul); BALL LOST / PASS flagged OFFSIDE (at the end of the pass); BALL LOST
                    FORCED, VOLUNTARY or END HALF (the referee halts play)
  restarts play     the first ball-playing event after the stoppage: a SET PIECE, or a RECOVERY
                    ("brings the ball back into play via a dead-ball situation that is not a SET
                    PIECE" - goal kicks, drop balls)
  also              a SET PIECE with no tagged stoppage before it: dead from the end of the last
                    ball event before it; D before each period's kick-off and after its end

Graded frames: each period's tracking from its first frame to its last event, as in the
challenge's ground truth (D before kick-off).

verify() checks the keys against the event log they come from, event by event.
"""

import numpy as np

PLAYS = {"PASS", "CARRY", "SHOT", "SET PIECE", "BALL LOST", "RECOVERY"}
NO_CHANGE = {"CHALLENGE", "FAULT RECEIVED", "CARD", "BALL OUT"}


def stop_frame(e):
    """Frame at which this event stops play, or None."""
    t, s = e["type"], e["subtype"].upper()
    end = max(e["start"], e["end"])
    if t == "BALL OUT":
        return end
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return end
    if t == "FAULT RECEIVED":
        return e["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return e["start"]
    if t in ("BALL LOST", "PASS") and "OFFSIDE" in s:
        return end
    if t == "BALL LOST" and any(k in s for k in ("FORCED", "VOLUNTARY", "END HALF")):
        return end
    return None


def restart(e):
    """A set piece, or a goal kick (kept for callers that simulate set-piece tags)."""
    return e["type"] == "SET PIECE" or "GOAL KICK" in e["subtype"]


def answer_key(G):
    """(label (n,) int8: 0 H, 1 A, 2 D, -1 not graded; dead (n,) bool; graded (n,) bool)"""
    n = len(G["period"])
    ev = sorted(G["events"], key=lambda e: (e["start"], e["end"]))
    graded = np.zeros(n, bool)
    period_end = {}
    for h in (1, 2):
        evh = [e for e in ev if e["period"] == h]
        a = int(np.flatnonzero(G["period"] == h)[0])          # D until the kick-off
        b = max(max(e["end"], e["start"]) for e in evh)
        graded[a:b + 1] = True
        period_end[h] = b + 1

    # who last played the ball
    moments = []
    for e in ev:
        if e["type"] in PLAYS and e["team"] is not None:
            moments.append((e["start"], e["team"]))
            if e["type"] == "PASS" and e["end"] > e["start"] and "OFFSIDE" not in e["subtype"].upper():
                moments.append((e["end"], e["team"]))
    moments.sort()
    team = np.full(n, -1, np.int8)
    for (f, t), (f2, _) in zip(moments, moments[1:] + [(n, None)]):
        team[f:f2] = t

    # dead spells: each stoppage until the next ball-playing event
    dead = np.zeros(n, bool)
    plays = [e for e in ev if e["type"] in PLAYS]
    starts = np.array([e["start"] for e in plays])
    for e in ev:
        s = stop_frame(e)
        if s is None:
            continue
        nxt = np.flatnonzero(starts > s)
        nxt = [plays[i] for i in nxt if plays[i]["period"] == e["period"]]
        r = nxt[0]["start"] if nxt else period_end[e["period"]]
        dead[s + 1:r] = True
    # set pieces with no tagged stoppage before them
    for i, e in enumerate(ev):
        if e["type"] != "SET PIECE":
            continue
        prev = [p for p in ev[:i] if p["period"] == e["period"] and p["type"] in PLAYS | {"BALL OUT"}
                and p["start"] < e["start"]]
        if not prev:                                          # the period's first kick-off
            continue
        stop = max(prev[-1]["start"], prev[-1]["end"])
        if not dead[min(n - 1, e["start"] - 1)] and e["start"] - stop > 1:
            dead[stop + 1:e["start"]] = True
    for h, b in period_end.items():                           # after the half's last event
        idx = np.flatnonzero(G["period"] == h)
        dead[b:idx[-1] + 1] = True
    label = np.where(dead, 2, team).astype(np.int8)
    label[~graded] = -1
    label[graded & (label == -1)] = 2
    return label, dead & graded, graded


def verify(G, label):
    """Check the key against the event log, event by event. Returns a dict of counts."""
    R = {"ball_events": 0, "team_matches": 0, "in_dead": 0, "stoppages": 0, "dead_after_stop": 0,
         "changes": 0, "changes_at_ball_event": 0}
    for e in G["events"]:
        if e["type"] in PLAYS and e["team"] is not None:
            R["ball_events"] += 1
            f = e["start"]
            # the frame after the event starts: a restart's own frame closes the stoppage
            v = label[min(len(label) - 1, f)]
            R["team_matches"] += int(v == e["team"])
            R["in_dead"] += int(v == 2)
        s = stop_frame(e)
        if s is not None:
            R["stoppages"] += 1
            R["dead_after_stop"] += int(label[min(len(label) - 1, s + 1)] == 2)
    # every possession change must sit on a ball-playing event of the new team
    starts = {}
    for e in G["events"]:
        if e["type"] in PLAYS and e["team"] is not None:
            starts.setdefault(e["start"], set()).add(e["team"])
            if e["type"] == "PASS":
                starts.setdefault(e["end"], set()).add(e["team"])
    ha = np.isin(label, [0, 1])
    for i in np.flatnonzero(ha[1:] & ha[:-1] & (label[1:] != label[:-1])) + 1:
        R["changes"] += 1
        R["changes_at_ball_event"] += int(label[i] in starts.get(int(i), set()))
    return R


def to_letters(label):
    return np.where(label == 0, "H", np.where(label == 1, "A", np.where(label == 2, "D", "X")))
