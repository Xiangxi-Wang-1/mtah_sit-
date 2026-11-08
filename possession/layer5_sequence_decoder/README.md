# Layer 5 · Sequence decoder (event-anchored, semi-Markov)

    python -m possession.layer5_sequence_decoder.run               # ~1 s -> cache/layer5_tactical.npz
    python -m possession.layer5_sequence_decoder.evaluate --sweep  # checks + settings sweep

Turns everything below it into one clean H / A / D timeline per half.

## v1 (current): driven by the cross-game models

Layer 5 now uses the layer-4 cross-game models: 0.35 × logit(`p_home_full`) per frame, plus at
every learned change point, 1.0 × log(P(change to that team) / P(no change)), added to the event
bonuses below. These settings were chosen on the bench by coordinate descent (`SEQ_W_*` in
config.py). The search used about 150 evaluations and accepted a change only if it gained at least
0.003; all three bench games improved over v0:

| bench spellF1 | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| v0 | 0.535 | 0.501 | 0.489 | 0.508 |
| cross-game frame model only (weight 0.2) | 0.678 | 0.630 | 0.627 | 0.645 |
| + learned change points, weight 0.35 (v1) | **0.684** | **0.682** | **0.651** | **0.672** |

What is still missed: possessions under 1 s are almost never found (5%), and 1-3 s possessions
39%. From 10 s up, 87-90% are matched. The bench keys have many more sub-second possessions than
the public window (14% vs 1 in 17), so part of that is Metrica's tagging style. Settings that
create more possessions scored worse on the bench.

## How it works (v0; the structure is unchanged)

- **Dead ball** comes fixed from layer 3. Each spell of live play is decoded on its own.
- **Candidate change points** - possession may change only here: the start of every reception, every
  kick (a player plays the ball away first time - new in layer 2), every touch attributed to a
  player, every frame the ball re-appears, and one point a second inside ball gaps over 2 s.
- **Evidence per frame** (log-odds that home has the ball): controls from layer 2 (1 nat a frame,
  less if contested), plus the players-only possession model from layer 4 (0.02 × its log-odds a
  frame - frames are strongly correlated, so each counts for little).
- **Semi-Markov dynamic programming**: a possession is a run of stretches with one team; its score
  is its evidence, plus a log-normal prior on its length, minus a switch cost at its start that a
  reception (8 nats) or kick (12 nats) by the new team pays back. The best sequence for the spell is
  found exactly, in about a second for the whole match.

## What the challenge counts as a possession

The public window (CHALLENGE.md: "use it to understand how these cases are labelled") has
possessions of 0.8 s and 1.3 s, and gives the ball to the team that last played it - including a
first-time clearance - until the other team controls it. So a reception or a kick by the other team
is a possession however short. Two consequences:

- layer 2 now detects **kicks** (the ball accelerating away from a player within 1.5 m who touched it
  only briefly) - 69 in the match, kicker a median 0.76 m from the ball;
- the settings are chosen by an objective that requires receptions and kicks to be honoured, not by
  agreement with Smart Tagging alone (its long phases contain no brief possessions, so it rewards
  merging them).

## Settings (no ground truth used to choose)

Swept 192 combinations; objective = Smart Tagging agreement (all + stale frames) + turnovers found +
consistency with certain moments + receptions honoured + kicks honoured, with a floor of 2
possessions a minute. The top ten are within noise of each other; their public-window spellF1
ranges from 0.45 to 0.55, which is how much that window can tell settings apart. The first-ranked
setting is used.

## Result

| | full match, no ground truth | |
|---|---|---|
| Smart Tagging agreement | 0.892 | (layers 0–3: 0.876) |
| ... on stale stretches | 0.944 | (layers 0–3: 0.69) |
| Smart Tagging turnovers found | 24 / 24 | |
| certain moments decoded to the right team | 99.9% | |
| uncontested receptions honoured | 99.2% | |
| kicks honoured | 93% | |
| possessions per minute | 2.64 | (football: ~2–4) |

Public 5 minutes, check only: spellF1 **0.545** (baseline 0.326), macroF1 0.830, 16 possessions
(truth 17).

## What is left

The remaining errors in the public window are mostly upstream: possessions with no detected
reception or kick (layer 2), and stoppage boundaries a few seconds off (layer 3). The decoder can
only switch at an event it is given.
