"""Layer 1 - physics features per frame.

    python -m possession.layer1_features.run        # writes cache/layer1_tactical.npz

Input : cache/layer0_tactical.npz, data/tactical/*_smart_tagging.xml
Output: one array per feature, indexed by video frame. The list and meaning of every feature is
in features.py and README.md. Gitignored derived data - never commit the cache.
"""

import json
import os
import time
import warnings

import numpy as np

from ..common import ROOT, cache_path, load_layer, smart_tagging_arrays
from .features import build_features

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True):
    t0 = time.time()
    z = load_layer("layer0")
    meta0 = json.loads(str(z["meta"]))
    home, away = meta0["teams"]
    st = smart_tagging_arrays(len(z["half"]), home, away)
    F, _ = build_features(z, st)
    if verbose:
        print("{} features  ({:.0f}s)".format(len(F), time.time() - t0))
    return F


def main():
    F = build()
    out = cache_path("layer1")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                               for k, v in F.items()})
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
