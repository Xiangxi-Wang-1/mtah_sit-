# Layer 1 · Physics features per frame

    python -m possession.layer1_features.run        # ~3 s  -> cache/layer1_tactical.npz
    python -m possession.layer1_features.evaluate   # ~10 s -> reports/layer1/report.md

65 features per video frame from the layer-0 arrays plus the Smart Tagging. Team features come in
`_h` / `_a` pairs; positions along the pitch are in each team's attacking frame (0 = own goal
line), so both halves read the same. `features.RECOMMENDED` lists the set later layers should use.

| group | features | what they say |
|---|---|---|
| ball-player | `d_*`, `margin`, `relv_*`, `closing_*`, `n2_*`, `n5_*`, `dwell_*`, `nearest_fwd_speed_*` | who is at the ball, moving with it, crowding it |
| multi-scale | `margin_1s/3s/10s`, `near_share_1s/3s/10s`, `margin_before_gap`, `margin_after_gap` | the same evidence over time, and carried across ball gaps |
| ball direction | `ball_x_home_frame`, `ball_ahead_*`, `ball_fwd_speed_home`, `ball_fwd_1s` | where the ball is and where it is going, relative to each team |
| team shape | `territory`, `spread_x_*`, `spread_y_*`, `area_*`, `centroid_fwd_speed_*` | which team is spread out to attack - needs no ball |
| dead-ball cues | `ball_outside`, `ball_to_line`, `ball_to_restart_spot`, `ball_still_frames`, `mean_speed_*`, gap context | for layer 4 |
| Smart Tagging | `st_possessor`, `st_set_piece`, `st_set_piece_team` | automatic phases and set pieces |

## How it is evaluated

No ground truth. The yardstick is Metrica's automatic Smart Tagging: its phase codes imply a side in
possession on 39% of graded frames (`common.POSSESSOR`, semantics checked against the data). It is
coarse, so it is used to compare features, not as a score to chase.

| evidence | AUC vs proxy | ball visible | ball missing |
|---|---|---|---|
| ball features, this frame only | 0.83 | 0.87 | - |
| ball features, multi-scale | 0.88 | 0.91 | 0.68 |
| team shape (5 features) | 0.92 | 0.92 | 0.94 |
| recommended set | 0.93 | 0.93 | 0.94 |

The best single feature is `area_diff` (0.90): the side with the ball spreads out.

**Timing matters as much as AUC**, because the competition scores whole possessions. At the 24
turnovers the Smart Tagging marks, shape features switch sides 6–10 s late (a team takes that
long to reshape), while `n2` (players within 2 m) and `dwell` switch within ~0.2 s. So layer 5 takes
**timing from the ball features and identity from shape** - neither alone is enough.

Redundant and left out of `RECOMMENDED`: centroid / defensive line / front line (|r| > 0.95 with
`territory`).
