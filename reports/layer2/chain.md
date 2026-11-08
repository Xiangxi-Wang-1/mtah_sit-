# Layers 0-2 together

How well do layers 0-2 work together? A minimal possession read-out from the touches.

    python -m possession.layer2_touches.evaluate_chain

Not the final decoder (that is layer 5), just enough to see whether the touch layer carries the
right information. Rule: possession changes only when a player of the other team controls the
ball (a control segment of at least MIN_CONTROL frames); everything else - deflections, loose
balls, gaps - keeps the last team. Compared against the official baseline (nearest player,
+/- 50 frame majority vote) on three things that need no ground truth:

  - agreement with the Smart Tagging possession proxy (frames where a phase implies a side)
  - timing at the Smart Tagging turnovers (start of each ATTACKING TRANSITION)
  - possessions per minute and their length (football: roughly 2-4 per minute of play)

```
{
 "n_turnovers": 24,
 "baseline_smooth50": {
  "proxy_agreement": 0.7222253181016067,
  "turnovers_found_within_10s": 1.0,
  "turnover_lag_median_s": -2.62,
  "turnover_abs_lag_median_s": 3.5,
  "possessions_per_min": 4.881748994378592,
  "possession_s_median": 4.92
 },
 "touch_readout_min_control_8": {
  "proxy_agreement": 0.8175545623593123,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -3.92,
  "turnover_abs_lag_median_s": 6.16,
  "possessions_per_min": 2.0018003613312603,
  "possession_s_median": 15.14
 },
 "touch_readout_min_control_12": {
  "proxy_agreement": 0.8181580921932601,
  "turnovers_found_within_10s": 0.875,
  "turnover_lag_median_s": 1.12,
  "turnover_abs_lag_median_s": 5.44,
  "possessions_per_min": 1.5485625436713522,
  "possession_s_median": 21.44
 },
 "touch_readout_min_control_20": {
  "proxy_agreement": 0.8424297784882393,
  "turnovers_found_within_10s": 0.8333333333333334,
  "turnover_lag_median_s": 2.48,
  "turnover_abs_lag_median_s": 5.84,
  "possessions_per_min": 1.2086341804264211,
  "possession_s_median": 32.56
 },
 "readout_contested>0.5_needs_20": {
  "proxy_agreement": 0.8393142596156983,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -2.94,
  "turnover_abs_lag_median_s": 5.36,
  "possessions_per_min": 1.8884909069162832,
  "possession_s_median": 15.36
 },
 "readout_contested>0.5_needs_40": {
  "proxy_agreement": 0.8375199817309888,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -2.94,
  "turnover_abs_lag_median_s": 5.36,
  "possessions_per_min": 1.8696059978471202,
  "possession_s_median": 16.6
 },
 "readout_contested>0.25_needs_25": {
  "proxy_agreement": 0.8428865037679836,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -2.94,
  "turnover_abs_lag_median_s": 5.36,
  "possessions_per_min": 1.8507210887779575,
  "possession_s_median": 16.86
 },
 "readout_contested>0.0_needs_25": {
  "proxy_agreement": 0.8445176654813559,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -2.94,
  "turnover_abs_lag_median_s": 5.66,
  "possessions_per_min": 1.8507210887779575,
  "possession_s_median": 16.86
 },
 "touch_readout_plus_close_touches": {
  "proxy_agreement": 0.7526995726356311,
  "turnovers_found_within_10s": 0.9166666666666666,
  "turnover_lag_median_s": -2.94,
  "turnover_abs_lag_median_s": 6.26,
  "possessions_per_min": 2.209534361092051,
  "possession_s_median": 15.28
 }
}
```
