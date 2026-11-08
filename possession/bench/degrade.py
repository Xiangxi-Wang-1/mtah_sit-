"""Make a bench game look like the challenge's tracking, so a score on it means something.

Measured on the challenge match (cache/layer0_tactical.npz, graded frames) vs the bench:

                                    challenge          bench (as delivered)
  ball seen while alive / dead      97.8% / 24%        96-98% / 1-15%     -> left as it is
  ball speed p50 / p90 / p99        5 / 14 / 25 m/s    5-7 / 15 / 21-24   -> left as it is
  players seen per team             10.2-10.3 of 11    11.0               -> dropouts
  player tracks (slot runs) / min   11.5               ~0 (one per player) -> fragmentation
  cross-team slot mix-ups           66 repaired        none               -> mix-ups

So the ball is not touched, and the players get:
  dropouts       each player is lost for a while every ~100 s (log-normal, median 4 s), which
                 leaves ~6% of player-frames missing
  fragmentation  every time a player is re-acquired he comes back in a new slot (the lowest free
                 slot of his team), as an optical tracker does
  mix-ups        6% of re-acquisitions start in a slot of the *other* team, for a log-normal
                 ~3 s, then hand back to a slot of his own team - the 12-vs-10 episodes layer 0's
                 team repair exists for
  jitter         0.1 m white noise on every position

and Smart-Tagging-like set-piece windows are simulated from the answer key's restarts: 95% of
restarts get one, starting a log-normal ~7 s (2-25 s) before the restart, 35 s long (5 s for a
kick-off), tagged with the restarting team except for throw-ins (the real tagging says N/A).
Measured on the challenge match: 56/59 set pieces tagged, the restart a median 7 s into a window
of median 35 s.

Everything is seeded per game, so the bench is reproducible.
"""

import numpy as np

SET_PIECE_TYPES = ["Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)",
                   "Kick Off (Start)"]
FPS = 25


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def players(P, team_of_slot, seed, on_mean_s=100.0, off_median_s=4.0, p_mix=0.06,
            mix_median_s=3.0, jitter_m=0.1, pitch=(105.0, 68.0)):
    """(n, k, 2) one column per real player -> (n, slots, 2) tracker-like slots, team_of_slot."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    pieces = []                                     # (start, end, true column, slot team)
    for k in range(P.shape[1]):
        t = int(team_of_slot[k])
        for a, b in _runs(~np.isnan(P[:, k, 0])):
            f = a
            first = True
            while f < b:
                on = max(25, int(rng.exponential(on_mean_s) * FPS))
                e = min(b, f + on)
                if not first and rng.random() < p_mix:   # re-acquired under the wrong team
                    m = min(e, f + max(5, int(rng.lognormal(np.log(mix_median_s), 0.7) * FPS)))
                    pieces.append((f, m, k, 1 - t))
                    if m < e:
                        pieces.append((m, e, k, t))
                else:
                    pieces.append((f, e, k, t))
                first = False
                f = e + int(rng.lognormal(np.log(off_median_s), 1.0) * FPS)
    pieces.sort()
    out_cols, out_team = [], []
    free_at = {0: [], 1: []}                        # per team: [(frame slot becomes free, slot)]
    for a, b, k, t in pieces:
        slots = free_at[t]
        free = [i for i, (ff, s) in enumerate(slots) if ff <= a]
        if free:
            i = min(free, key=lambda i: slots[i][1])
            s = slots[i][1]
            slots[i] = (b, s)
        else:
            s = len(out_cols)
            out_cols.append(np.full((n, 2), np.nan))
            out_team.append(t)
            slots.append((b, s))
        out_cols[s][a:b] = P[a:b, k]
    Q = np.stack(out_cols, axis=1)
    Q = Q + rng.normal(0, jitter_m, Q.shape) / np.array(pitch)
    return Q, np.array(out_team, np.int8)


def set_piece_tags(events, n, seed, restart_fn, p_tag=0.95):
    rng = np.random.default_rng(seed + 1)
    sp = np.zeros(n, np.int8)
    sp_team = np.full(n, -1, np.int8)
    started = set()
    for e in sorted(events, key=lambda e: e["start"]):
        if not restart_fn(e) or rng.random() > p_tag:
            continue
        st = e["subtype"].upper()
        if e["type"] != "SET PIECE":
            kind = "Goal Kick"
        elif "THROW" in st:
            kind = "Throw-in"
        elif "CORNER" in st:
            kind = "Corner Kick"
        elif "KICK OFF" in st:
            kind = "Kick Off (After Goal)" if e["period"] in started else "Kick Off (Start)"
            started.add(e["period"])
        else:
            kind = "Free Kick"
        lead = np.clip(rng.lognormal(np.log(7.0), 0.6), 2, 25)
        length = 5.0 if "Kick Off" in kind else 35.0
        a = max(0, int(e["start"] - lead * FPS))
        b = min(n, int(a + length * FPS))
        sp[a:b] = 1 + SET_PIECE_TYPES.index(kind)
        if kind != "Throw-in" and e["team"] is not None:
            sp_team[a:b] = e["team"]
    return {"st_possessor": np.zeros(n, np.int8), "st_set_piece": sp, "st_set_piece_team": sp_team}
