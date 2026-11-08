"""Repair cross-team identity switches with the 11-a-side constraint.

The tracker's team is per track slot, but a slot can drift onto an opponent during a duel and
keep following him. That leaves frames where one team shows 12+ players and the other 10 or
fewer, while the total is still ~22. The count says that one slot of the over-full team is
wrong; the positions say where it could have gone wrong.

Greedy repair. On each 12-vs-10 episode, every slot of the over-full team is a suspect, and
every moment it could have switched person is a candidate switch point:

  jump     the slot's position jumped faster than a human can run (players.py splits there)
  contact  an opponent's track ended within SWITCH_RADIUS of the slot: the tracker plausibly
           lost the opponent because this slot took him over
  start    the slot's run began shortly before the episode (TEAM_REPAIR_LEAD_IN); stronger
           when it was born where an opponent's track had just been lost (the tracker
           re-acquired that opponent under the wrong team)

Relabelling a suspect from a switch point to the end of its run is scored by how many excess
players it removes, minus a penalty for every frame where it pushes the other team over 11.
Ties go to the stronger evidence (jump > contact whose motion continues the opponent's >
re-acquisition start > other contact > start). The best change is kept only if it removes at least half the episode.
"""

import numpy as np

from .. import config as C

MAX_TEAM = 11


def _runs_1d(mask):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _excess(c):
    return np.maximum(0, c - MAX_TEAM)


def _predict(xy, vel, x, f, frames, k=5, horizon=25):
    """Where slot x would be at `frames` had it kept its velocity from just before f."""
    with np.errstate(all="ignore"):
        v = np.nanmean(vel[max(0, f - k):f + 1, x], axis=0)
    if np.isnan(v).any():
        v = np.zeros(2)
    dt = np.minimum(np.asarray(frames) - f, horizon)[:, None] / C.FPS
    return xy[f, x][None, :] + v[None, :] * dt


def _swap_vs_birth(xy, vel, s, o, f, run_end_s, run_start_row, newborn, k=4):
    """Weigh two explanations of opponent o's track ending next to slot s at frame f.

    swap  : the tracker moved s onto o. Then s, after f, follows o's predicted path, and the
            player s used to follow re-appears later in a new slot near s's own predicted path.
    birth : the tracker just lost o, and later re-acquired him in a new slot of the wrong
            team. Then s carries on along its own path, and the new slot appears near o's.
    Returns {"swap": metres of evidence for swap (negative favours birth), "newborn": slot}.
    """
    after = np.arange(f + 1, min(f + 1 + k, run_end_s))
    score = 0.0
    if len(after) >= 3:
        p = xy[after, s]
        e_self = np.nanmean(np.hypot(*(p - _predict(xy, vel, s, f, after)).T))
        e_opp = np.nanmean(np.hypot(*(p - _predict(xy, vel, o, f, after)).T))
        if np.isfinite(e_self) and np.isfinite(e_opp):
            score = float(e_self - e_opp)
    best_n, best_d = None, None
    for n in newborn:
        rs = run_start_row[n]
        if rs <= f or rs - f > 75:
            continue
        pn = xy[rs, n]
        d_own = np.hypot(*(pn - _predict(xy, vel, s, f, [rs])[0]))
        d_opp = np.hypot(*(pn - _predict(xy, vel, o, f, [rs])[0]))
        d = float(d_opp - d_own)  # > 0: newborn is where s's old player would be -> swap
        if best_d is None or abs(d) > abs(best_d):
            best_n, best_d = n, d
    if best_d is not None:
        score += best_d
    return {"swap": score, "newborn": best_n}


def repair_team_labels(xy, valid, team_of_slot, active, vel=None, jump_start=None,
                       verbose=False):
    """Returns (team (n_frames, n_slots) int8 with -1 where not valid, list of repairs).

    active     bool per frame: where the 11-a-side constraint applies (graded frames)
    vel        player velocities (m/s); finite differences of xy if not given
    jump_start bool (n_frames, n_slots): frames where a slot jumped (players.py)
    """
    n_frames, n_slots = valid.shape
    if vel is None:
        vel = np.gradient(xy, 1.0 / C.FPS, axis=0)
    if jump_start is None:
        jump_start = np.zeros_like(valid)
    team = np.where(valid, team_of_slot[None, :], -1).astype(np.int8)

    run_start = np.full((n_frames, n_slots), -1, dtype=np.int64)
    run_end = np.full((n_frames, n_slots), -1, dtype=np.int64)
    last_frames = []
    for s in range(n_slots):
        a, b = _runs_1d(valid[:, s])
        for x, y in zip(a, b):
            run_start[x:y, s] = x
            run_end[x:y, s] = y
        last_frames.append(b - 1)
    jumps = [np.flatnonzero(jump_start[:, s] & valid[:, s]) for s in range(n_slots)]

    def counts():
        return np.stack([(team == t).sum(axis=1) for t in (0, 1)], axis=1)

    repairs = []
    for _pass in range(C.TEAM_REPAIR_PASSES):
        c = counts()
        over = np.full(n_frames, -1, dtype=np.int8)
        over[(c[:, 0] > MAX_TEAM) & (c[:, 1] < MAX_TEAM) & active] = 0
        over[(c[:, 1] > MAX_TEAM) & (c[:, 0] < MAX_TEAM) & active] = 1
        accepted = 0
        for t in (0, 1):
            ea, eb = _runs_1d(over == t)
            for a, b in zip(ea, eb):
                if b - a < C.TEAM_REPAIR_MIN_EPISODE:
                    continue
                c = counts()
                best = None
                suspects = np.flatnonzero(team[a] == t)
                # contact events: (slot s, opponent o, frame f = o's last frame)
                events = []
                for s in suspects:
                    rs = run_start[a, s]
                    for o in range(n_slots):
                        if o == s:
                            continue
                        lf = last_frames[o]
                        for f in lf[(lf >= rs) & (lf <= a)]:
                            if team[f, o] == 1 - t and \
                                    np.hypot(*(xy[f, o] - xy[f, s])) <= C.SWITCH_RADIUS:
                                events.append((s, o, int(f)))
                newborn = [n for n in suspects if a - run_start[a, n] <= C.TEAM_REPAIR_LEAD_IN]
                cand = {s: {} for s in suspects}

                def offer(s, f, ev):
                    cand[s][f] = max(cand[s].get(f, -np.inf), ev)

                for n in newborn:
                    offer(n, run_start[a, n], -5.0)
                for s in suspects:
                    rs = run_start[a, s]
                    for f in jumps[s][(jumps[s] > rs) & (jumps[s] <= a)]:
                        offer(s, int(f), 10.0)
                for s, o, f in events:
                    # any suspect born shortly after o was lost can be o re-acquired
                    j_ = _swap_vs_birth(xy, vel, s, o, f, run_end[a, s], run_start[a],
                                        [n for n in suspects if n != s])
                    offer(s, f + 1, j_["swap"])
                    if j_["newborn"] is not None:
                        offer(j_["newborn"], run_start[a, j_["newborn"]], -j_["swap"])
                for s in suspects:
                    re_ = run_end[a, s]
                    cands = cand[s]
                    for f, ev in cands.items():
                        span = slice(f, re_)
                        ct, co = c[span, t], c[span, 1 - t]
                        act = active[span]
                        gain = int(np.sum(act * ((_excess(ct) - _excess(ct - 1))
                                                 - (_excess(co + 1) - _excess(co)))))
                        created = int(np.sum(act * (_excess(co + 1) > _excess(co))))
                        key = (gain - 2 * created, ev, -(re_ - f))
                        if best is None or key > best[0]:
                            best = (key, s, f, re_, gain, created, ev)
                if best is None:
                    continue
                (key, s, f, re_, gain, created, ev) = best
                if key[0] >= max(C.TEAM_REPAIR_MIN_GAIN, 0.5 * (b - a)):
                    team[f:re_, s] = 1 - t
                    kind = ("jump" if ev >= 10 else "start" if ev <= -5 else
                            "contact" if f != run_start[a, s] else "reacquire")
                    repairs.append({"slot": int(s), "from": int(f), "to": int(re_),
                                    "new_team": int(1 - t), "gain": gain, "created": created,
                                    "evidence": kind, "score": round(float(ev), 3)})
                    accepted += 1
        if verbose:
            print("team repair pass", _pass, "accepted", accepted)
        if accepted == 0:
            break
    return team, repairs
