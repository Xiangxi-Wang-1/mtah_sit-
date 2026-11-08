# Layer 4 · Possession models (ML)

    python -m possession.layer4_possession_model.run       # -> cache/layer4_tactical.npz
    python -m possession.layer4_possession_model.evaluate  # -> reports/layer4/report.md

Two kinds of model:

- **Cross-game models** (`crossgame.py`, used by layer 5). These are trained on every live frame of
  the three bench games, which have true labels (see `possession/bench`). They are never trained on
  the game they predict.
- **Self-trained model** (`model.py`). This learns on this match from moments where possession is
  certain, without any human label. Layer 5 no longer uses it; it stays as a reference.

The first run needs the bench: `python -m possession.bench.build`, then
`python -m possession.bench.score`, which runs layers 1-3 on every game before layer 4. Trained
models and feature matrices are cached in `cache/models/` (gitignored). Training takes ~3 min
per model on 2 cores.

## Cross-game models (v1, the ones layer 5 uses)

| output | what it is |
|---|---|
| `p_home_full` | P(home has the ball) on every frame. Features: players (shape, pressing, movement, line positions, sprinting and retreating shares) plus ball (distances, pressure, who controlled it last and who controls it next), averaged over windows from 8 s before to 8 s after the frame |
| `cp_frames`, `cp_p` | change points: every third live frame, P(no change / change to home / change to away within ±6 frames). Same features plus layer-2 events of each team just before, at and after the frame |

Each training row is also added with home and away swapped and the label flipped. Each prediction
is the average over both orientations, so the model learns "the team with the ball" and not one
particular home team.

| bench (leave one game out) | v0: self-trained players model | v1: cross-game |
|---|---|---|
| live frames with the right team, per frame | ~0.88 | **0.946 / 0.955 / 0.945** |
| pipeline spellF1 (mean of 3) | 0.508 | **0.672** |

On our match, measured on the certain moments, the cross-game model is right 98.3% of the time;
the self-trained players model is right 92.7%. On the bench, those certain moments are 98-99%
correct against the truth. Smart Tagging agreement drops from 0.914 to 0.881. Smart Tagging lags
and merges brief possessions, so it rewards slow signals, and we don't treat it as the judge here.

## Self-trained model (v0, kept for reference)

### Labels: moments where possession is certain

A player controlling the ball (layer 2) for ≥ 1 s with no opponent within reach, in play (layer 3),
plus the ball travelling between two such controls of the same team within 3 s (a completed pass).
27,207 frames in 600 spells, 67% Barcelona. An algorithm picks them, so the rules allow it.

## Model

Gradient boosting (`HistGradientBoostingClassifier`, 31 leaves). Every prediction is out-of-fold:
the match is cut into 10 contiguous time blocks, each predicted by a model that never saw it (nor
10 s either side), so the output is honest everywhere.

| output | features | use |
|---|---|---|
| `p_home_players` | players only: shape, pressing, movement, at 3 s / 10 s (and a context model with 30 s and ±5 s); even blend of the two | **the evidence through ball gaps and stale stretches** |
| `p_home_full` | players + ball | mostly relearns "nearest player" (the labels come from it); useful only for timing |

## Results (no ground truth used)

| | players view | full view | layers 0–3 read-out |
|---|---|---|---|
| held-out certain moments, accuracy | **92.7%** (AUC 0.964) | 99.5% (circular) | - |
| Smart Tagging agreement, all frames | **91.4%** | 76.3% | 86.2% |
| Smart Tagging agreement, stale frames* | **89.3%** | 56.8% | 68.9% |
| right where the read-out is wrong | 67% | 39% | - |
| lag at Smart Tagging turnovers | 2.5 s | 0.8 s | - |

\*ball missing, or no control for > 10 s.

Variants tried (`evaluate.py`, `VARIANTS`): labels without passes, tree size, a context-only model.
Context alone is best on held-out certain moments but worse on stale frames; the 50/50 blend beats
both on every measure, so it is the output.

A crude fusion (the model's answer only on stale stretches) moves the Smart Tagging agreement from
0.876 to 0.882 and leaves the public window unchanged, which has almost no stale stretches. The
real use is in layer 5, where switches can only happen at events.
