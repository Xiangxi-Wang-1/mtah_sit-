# possession — our pipeline

One folder per layer. Each layer reads the previous layer's output from `cache/` and writes its own.

| folder | layer | status |
|---|---|---|
| `layer0_clean/` | 0 · Clean the data | built |
| `layer1_features/` | 1 · Physics features per frame | built |
| `layer2_touches/` | 2 · Touches and ball-with-player segments | built |
| `layer3_dead_ball/` | 3 · Dead-ball detector | built |
| `layer4_possession_model/` | 4 · Possession evidence model (ML, self-trained) | built |
| `layer5_sequence_decoder/` | 5 · Sequence decoder (event-anchored semi-Markov, H/A/D) | built |
| `layer6_output/` | 6 · Abstention, submission CSV, validation | not built |

Layers run in number order; each reads only lower-numbered layers' caches.

`config.py` holds every tunable constant. None is tuned on any ground truth.

`cache/` (frame-level arrays) is gitignored: it is derived tracking data and must never be
published. `reports/` holds aggregate evaluation output only.

Run from the repository root:

```
python -m possession.layer0_clean.run        # ~20 s, writes cache/layer0_tactical.npz
python -m possession.layer0_clean.evaluate   # ~15 s, writes reports/layer0/
python -m possession.layer1_features.run      # ~3 s
python -m possession.layer1_features.evaluate
python -m possession.layer2_touches.run       # ~1 s
python -m possession.layer2_touches.evaluate --sweep
python -m possession.layer2_touches.evaluate_chain
python -m possession.layer3_dead_ball.run      # ~1 s
python -m possession.layer3_dead_ball.evaluate --sweep
python -m possession.layer4_possession_model.run      # ~90 s
python -m possession.layer4_possession_model.evaluate
python -m possession.layer5_sequence_decoder.run       # ~1 s
python -m possession.layer5_sequence_decoder.evaluate --sweep
python -m possession.compare_models            # baseline vs our models -> reports/model_comparison.md
```

`common.py` holds what several layers share: cache paths, the Smart Tagging reader, pitch geometry.

Requires numpy, pandas, scipy, scikit-learn, matplotlib.
