"""Layer 5 - sequence decoder.

    python -m possession.layer5_sequence_decoder.run     # writes cache/layer5_tactical.npz

Input : cache/layer0, layer2, layer3, layer4
Output: label per video frame (0 home, 1 away, 2 dead / not graded). Gitignored.
"""

import os
import time
import warnings

import numpy as np

from ..common import ROOT, cache_path, load_layer
from .decoder import decode

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True):
    t0 = time.time()
    z, L2, L3, L4 = (load_layer(x) for x in ("layer0", "layer2", "layer3", "layer4"))
    lab = decode(z, L2, L3, L4)
    if verbose:
        g = z["half"] > 0
        print("H {:.1%}  A {:.1%}  D {:.1%} of graded  ({:.0f}s)".format(
            np.mean(lab[g] == 0), np.mean(lab[g] == 1), np.mean(lab[g] == 2), time.time() - t0))
    return lab


def main():
    lab = build()
    path = cache_path("layer5")
    np.savez_compressed(path, label=lab)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
