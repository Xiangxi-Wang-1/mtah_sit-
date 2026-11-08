# Model comparison

Compare the official baseline with our read-outs, model by model.

    python -m possession.compare_models

Models
  baseline     the official baseline_smooth50 (nearest player, +/- 2 s majority vote)
  layers 0-2   touch read-out (possession changes only when the other team controls the ball;
               contested control must last 1 s); D wherever the ball is missing
  layers 0-5 v0  the full pipeline before the bench: event-anchored semi-Markov decoder (layer 5)
               over receptions, kicks and the self-trained possession model, stoppages from layer 3
  layers 0-5 v1  the same decoder over the cross-game models (layer 4, trained on the three bench
               games with true labels): P(home has the ball) per frame and learned change points;
               decoder settings chosen on the bench
  layers 0-3   the same possession read-out, with stoppages from layer 3: D during each dead
               spell, and after each restart the side is the first team to control the ball

Scored two ways:
  public 5 min  the official scorer against the public ground truth. This is a correctness
                check, not a tuning set (RULES.md): nothing in the pipeline is chosen from it.
                17 possessions, so one possession moves spellF1 by ~6 points.
  full match    agreement with the Smart Tagging proxy on H/A frames, the Smart Tagging
                turnovers, and possessions per minute (football: ~2-4).

Writes reports/model_comparison.md.

| | baseline (smooth50) | layers 0-2 | layers 0-3 | layers 0-5 v0 | layers 0-5 v1 (current) |
|---|---|---|---|---|---|
| spellF1 (ranked on) | 0.326 | 0.286 | 0.425 | 0.545 | 0.562 |
| macroF1 (tie-break) | 0.621 | 0.646 | 0.814 | 0.830 | 0.848 |
| H/A frames with right team | 83.5% | 80.6% | 81.5% | 83.7% | 86.2% |
| D frames called dead | 21.4% | 33.6% | 86.3% | 86.3% | 86.3% |
| possessions (truth 17) | 26 | 18 | 15 | 16 | 15 |
| mean possession s (truth 14.2) | 10.8 | 14.7 | 15.3 | 14.4 | 15.3 |
| Smart Tagging agreement | 0.722 | 0.853 | 0.876 | 0.892 | 0.881 |
| turnovers found (of 24) | 100% | 92% | 96% | 100% | 96% |
| possessions per min (~2-4) | 5.61 | 3.17 | 2.16 | 2.64 | 2.79 |
| dead share (~50%) | 32.3% | 40.8% | 52.4% | 52.4% | 52.4% |

```
{
 "baseline (smooth50)": {
  "public_5min": {
   "spellF1": 0.3255813953488372,
   "macroF1": 0.6207331924027959,
   "team_right_on_HA_frames": 0.8348532581661416,
   "dead_right_on_D_frames": 0.2137508509189925,
   "possessions": 26,
   "mean_possession_s": 10.803076923076924
  },
  "full_match_proxy": {
   "proxy_agreement": 0.7222253181016067,
   "turnovers_found_within_10s": 1.0,
   "turnover_lag_median_s": -2.62,
   "turnover_abs_lag_median_s": 3.5,
   "possessions_per_min": 5.6088179935413605,
   "possession_s_median": 4.6,
   "dead_share": 0.32336000302158546
  }
 },
 "layers 0-2": {
  "public_5min": {
   "spellF1": 0.2857142857142857,
   "macroF1": 0.6457921492299044,
   "team_right_on_HA_frames": 0.8060023213397447,
   "dead_right_on_D_frames": 0.336283185840708,
   "possessions": 18,
   "mean_possession_s": 14.71111111111111
  },
  "full_match_proxy": {
   "proxy_agreement": 0.8530215914116843,
   "turnovers_found_within_10s": 0.9166666666666666,
   "turnover_lag_median_s": -2.94,
   "turnover_abs_lag_median_s": 5.36,
   "possessions_per_min": 3.1726647236193557,
   "possession_s_median": 7.04,
   "dead_share": 0.4078384962576405
  }
 },
 "layers 0-3": {
  "public_5min": {
   "spellF1": 0.425,
   "macroF1": 0.8144950608380999,
   "team_right_on_HA_frames": 0.8149560603548334,
   "dead_right_on_D_frames": 0.8631722260040844,
   "possessions": 15,
   "mean_possession_s": 15.328
  },
  "full_match_proxy": {
   "proxy_agreement": 0.8764795211934432,
   "turnovers_found_within_10s": 0.9583333333333334,
   "turnover_lag_median_s": 0.28,
   "turnover_abs_lag_median_s": 5.12,
   "possessions_per_min": 2.162322088419144,
   "possession_s_median": 9.24,
   "dead_share": 0.5240499316995789
  }
 },
 "layers 0-5 v0": {
  "public_5min": {
   "spellF1": 0.5454545454545455,
   "macroF1": 0.8298581159669703,
   "team_right_on_HA_frames": 0.8365113579837506,
   "dead_right_on_D_frames": 0.8631722260040844,
   "possessions": 16,
   "mean_possession_s": 14.37
  },
  "full_match_proxy": {
   "proxy_agreement": 0.8916208852561526,
   "turnovers_found_within_10s": 1.0,
   "turnover_lag_median_s": -0.82,
   "turnover_abs_lag_median_s": 5.2,
   "possessions_per_min": 2.6438872696827964,
   "possession_s_median": 7.48,
   "dead_share": 0.5240499316995789
  }
 },
 "layers 0-5 v1 (current)": {
  "public_5min": {
   "spellF1": 0.5625,
   "macroF1": 0.8481652181508018,
   "team_right_on_HA_frames": 0.8623777151384513,
   "dead_right_on_D_frames": 0.8631722260040844,
   "possessions": 15,
   "mean_possession_s": 15.328
  },
  "full_match_proxy": {
   "proxy_agreement": 0.8807896735003796,
   "turnovers_found_within_10s": 0.9583333333333334,
   "turnover_lag_median_s": 1.8,
   "turnover_abs_lag_median_s": 4.6,
   "possessions_per_min": 2.794966542236099,
   "possession_s_median": 6.2,
   "dead_share": 0.5240499316995789
  }
 }
}
```
