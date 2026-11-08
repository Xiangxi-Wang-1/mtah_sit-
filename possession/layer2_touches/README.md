# Layer 2 · Touch detection

    python -m possession.layer2_touches.run              # ~1 s -> cache/layer2_tactical.npz
    python -m possession.layer2_touches.evaluate --sweep # self-supervised checks
    python -m possession.layer2_touches.evaluate_chain   # layers 0-2 together

## What a touch looks like in this data

The provider smooths the ball on a ~0.5 s scale, so a kick is a speed bump, not a jump.
Velocity cross-correlation with the dribbling player peaks at lag 0, so ball and players are in
sync (the ~5-frame position offset is the ball running ahead of the dribbler, which is physical).

## Events

| kind | from | meaning |
|---|---|---|
| `receive` / `release` | control segments: nearest player within 1.5 m **and** moving with the ball (relative speed < 3 m/s) for ≥ 0.32 s | a player has the ball; start = reception, end = pass/shot |
| `touch` | outside control: a speed dip (2–8 m/s deep) or a 20–90° turn of a ball moving 4–14 m/s, with a player within 2.5 m | deflection, block, one-touch |
| `unattributed` | same, no player within 2.5 m | probably aerial or a bounce |
| `artifact` | near-reversal of a fast ball | the tracker, not a player (no closer to players than random) |
| `stop` | very deep speed dip | the ball coming to rest |
| `kick` | the ball's speed rises by ≥ 5 m/s within 0.5 s, starting next to a player (≤ 1.5 m) who touched it only briefly | first-time pass or clearance: that team last played the ball |
| `lost` / `found` | edges of ball gaps | a hand-over can hide in a gap |

Every control segment carries `duration` and `contest` (share of it with an opponent also within
1.5 m). Per frame: `ball_state` (0 unknown, 1 with a player, 2 travelling), `control_team`,
`last_control_team`, `frames_since_control`, `last_control_contest`.

## How it is evaluated (no ground truth)

- **Complete**: 84% of free-ball stretches are fully explained by detected events (no unexplained
  speed-up or turn).
- **Plausible**: 12.9 receptions per minute of visible ball; the next receiver is a team-mate 78% of
  the time (pass completion in top leagues: ~75–90%); median pass 14 m; median hold 0.9 s.
- **Precise**: free touches are only slightly closer to players than random free-ball frames, so
  they are weak evidence and must not flip possession on their own. Control segments are the
  strong signal.

## Layers 0-2 together (`evaluate_chain.py`)

Minimal read-out: possession changes only when the other team controls the ball; a contested
control must last ≥ 1 s. Against the official baseline (nearest player, ±2 s majority vote):

| | Smart Tagging agreement | turnovers found | possessions / min | median length |
|---|---|---|---|---|
| baseline_smooth50 | 0.722 | 100% | 4.9 | 4.9 s |
| touch read-out | 0.843 | 92% | 1.85 | 16.9 s |

The read-out has no dead-ball state yet (layer 3), so stoppages don't split possessions - that is
why it shows fewer than the 2–4 per minute football usually has. Where it is wrong the shape
features from layer 1 are right 63% of the time: the two streams complement each other, which is
what layers 4 and 5 combine.

Findings that shaped layers 0-2:
- The layer-0 team repair changes no reception: cross-team identity switches happen ~25 m from the
  ball. Ball-side evidence is robust to them.
- Control thresholds sit on a plateau (proxy agreement 0.82–0.86 across 36 settings); defaults are
  mid-plateau rather than tuned to the proxy.
