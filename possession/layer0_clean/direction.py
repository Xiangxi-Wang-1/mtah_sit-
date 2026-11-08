"""Which way each team attacks in each half.

The team's median x over a half sits in its own half of the pitch: a home side whose players
mostly stand at x < L/2 defends the x = 0 goal and attacks towards +x. Checked against the
goalkeeper, the player who spends the half closest to one goal line.
"""

import numpy as np

from .parse import HOME, AWAY


def attack_direction(xy, valid, team_of_slot, half, pitch_length):
    """{half: {"home": +1/-1, "away": +1/-1, "home_median_x", "away_median_x", "gk_check"}}.

    +1 means the team attacks towards x = pitch_length.
    """
    out = {}
    x = np.where(valid, xy[..., 0], np.nan)
    for h in sorted(set(np.unique(half)) - {0}):
        frames = half == h
        res = {}
        for name, team in (("home", HOME), ("away", AWAY)):
            team_x = x[frames][:, team_of_slot == team]
            med = float(np.nanmedian(team_x))
            res[name + "_median_x"] = med
            res[name] = 1 if med < pitch_length / 2 else -1
            # Goalkeeper check: the deepest player of the team on each frame should sit near
            # the goal the team defends.
            seen = ~np.isnan(team_x).all(axis=1)
            deepest = (np.nanmin(team_x[seen], axis=1) if res[name] == 1
                       else np.nanmax(team_x[seen], axis=1))
            res[name + "_gk_median_x"] = float(np.nanmedian(deepest))
        res["consistent"] = (res["home"] == -res["away"]
                             and abs(res["home_gk_median_x"] - (0 if res["home"] == 1
                                                                 else pitch_length)) < 20
                             and abs(res["away_gk_median_x"] - (0 if res["away"] == 1
                                                                 else pitch_length)) < 20)
        out[int(h)] = res
    return out
