"""Layer 2 - touch detection.

    python -m possession.layer2_touches.run        # writes cache/layer2_tactical.npz

Input : cache/layer0_tactical.npz, cache/layer1_tactical.npz
Output: events (structured array: frame, kind, team, slot, dist, speed_before, speed_after,
        turn, duration, contest; kinds in touches.KIND_NAMES), and per frame ball_state
        (0 unknown, 1 with a player, 2 travelling), control_team, last_control_team,
        frames_since_control, last_control_contest. Gitignored derived data.
"""

import os
import time
import warnings

import numpy as np

from ..common import ROOT, cache_path, load_layer
from .touches import KIND_NAMES, detect

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True):
    t0 = time.time()
    z = load_layer("layer0")
    F = {k: v for k, v in load_layer("layer1").items()}
    F = {k: (v.astype(float) if v.dtype.kind == "f" else v) for k, v in F.items()}
    E, frames = detect(z, F)
    if verbose:
        counts = {KIND_NAMES[k]: int((E["kind"] == k).sum()) for k in range(len(KIND_NAMES))}
        print("events", counts, " ({:.0f}s)".format(time.time() - t0))
    return E, frames


def main():
    E, frames = build()
    out = cache_path("layer2")
    np.savez_compressed(out, events=E, **frames)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
