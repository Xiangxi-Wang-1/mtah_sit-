"""Clean the player tracks: split identity switches, drop duplicates, smooth, differentiate.

Track slots are tracklets, not people (46 slots for 22 players). A slot can jump from one
person to another between frames; such a jump is impossible for a human, so the slot's run is
split there and velocities are never computed across it.
"""

import numpy as np
from scipy.signal import savgol_filter

from .. import config as C


def split_identity_switches(xy):
    """Boolean (n_frames, n_slots): True on frames where a slot starts a new run.

    A run starts after a missing frame or after a step faster than PLAYER_V_MAX.
    """
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1))  # (n_frames-1, n_slots)
    jump = step > C.PLAYER_V_MAX * C.DT + C.PLAYER_JITTER
    visible = ~np.isnan(xy[..., 0])
    start = np.zeros(visible.shape, dtype=bool)
    start[0] = visible[0]
    start[1:] = visible[1:] & (~visible[:-1] | jump)
    return start


def smooth_and_differentiate(xy, run_start, window=C.PLAYER_SG_WINDOW,
                             order=C.PLAYER_SG_ORDER):
    """SG-smoothed positions and velocities per slot, run by run."""
    n_frames, n_slots, _ = xy.shape
    pos = np.full_like(xy, np.nan)
    vel = np.full_like(xy, np.nan)
    visible = ~np.isnan(xy[..., 0])
    for s in range(n_slots):
        vis = visible[:, s]
        if not vis.any():
            continue
        starts = np.flatnonzero(run_start[:, s])
        # A run ends where the next starts or where the slot goes missing.
        ends_missing = np.flatnonzero(vis & ~np.concatenate((vis[1:], [False]))) + 1
        bounds = np.sort(np.concatenate((starts, ends_missing)))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if not vis[a]:
                continue
            seg = xy[a:b, s]
            n = b - a
            if n >= window:
                pos[a:b, s] = savgol_filter(seg, window, order, axis=0, mode="interp")
                vel[a:b, s] = savgol_filter(seg, window, order, deriv=1, delta=C.DT, axis=0,
                                            mode="interp")
            else:
                pos[a:b, s] = seg
                if n >= 2:
                    vel[a:b, s] = np.gradient(seg, C.DT, axis=0)
    return pos, vel


def duplicate_mask(xy, team_of_slot):
    """(n_frames, n_slots) True where a slot duplicates a lower-numbered slot of its team.

    Two same-team slots closer than DUPLICATE_RADIUS on one frame are one person tracked twice.
    The lower slot index is kept so the choice is deterministic.
    """
    n_frames, n_slots, _ = xy.shape
    dup = np.zeros((n_frames, n_slots), dtype=bool)
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            if team_of_slot[i] != team_of_slot[j]:
                continue
            d = np.hypot(*(xy[:, i] - xy[:, j]).T)
            dup[:, j] |= d < C.DUPLICATE_RADIUS  # NaN compares False
    return dup


def clean_players(xy_raw, team_of_slot):
    """The whole player step. Returns dict with xy, vel, valid, duplicate, run_start."""
    run_start = split_identity_switches(xy_raw)
    pos, vel = smooth_and_differentiate(xy_raw, run_start)
    dup = duplicate_mask(xy_raw, team_of_slot)
    valid = ~np.isnan(pos[..., 0]) & ~dup
    seen = ~np.isnan(xy_raw[..., 0])
    jump_start = run_start.copy()
    jump_start[0] = False
    jump_start[1:] &= seen[:-1]  # a run start right after a visible frame is a jump
    return {"xy": pos, "vel": vel, "valid": valid, "duplicate": dup, "run_start": run_start,
            "jump_start": jump_start}
