# Layer 1 evaluation

Proxy: Smart Tagging phases imply the side in possession on 39% of graded frames (61306 frames, home share 70%).

## Combined (logistic, blocked CV)

| group | features | AUC | ball visible | ball missing |
|---|---|---|---|---|
| ball_now | 11 | 0.828 | 0.870 | 0.311 |
| ball_multiscale | 21 | 0.876 | 0.905 | 0.680 |
| shape_full | 11 | 0.922 | 0.918 | 0.930 |
| shape_lean | 9 | 0.921 | 0.914 | 0.937 |
| all | 32 | 0.927 | 0.925 | 0.930 |
| recommended | 30 | 0.931 | 0.929 | 0.931 |

## Transition timing at 24 proxy hand-overs

Where a step from the old side to the new side best fits the evidence, relative to the proxy's hand-over (+ = later). The proxy's own boundaries are only good to ~1-2 s.

| evidence | median lag | median abs lag | within 2 s |
|---|---|---|---|
| model_ball_multiscale | +3.7 s | 5.1 s | 12% |
| model_shape_lean | +6.2 s | 7.3 s | 0% |
| model_recommended | +4.9 s | 6.4 s | 4% |
| margin | +1.4 s | 4.0 s | 30% |
| margin_1s | +1.5 s | 4.1 s | 25% |
| near_share_3s | +2.0 s | 5.0 s | 24% |
| ball_fwd_1s | +3.2 s | 3.9 s | 15% |
| dwell_diff | +0.1 s | 3.4 s | 33% |
| n2_diff | +0.2 s | 3.0 s | 29% |
| area_diff | +9.1 s | 10.0 s | 0% |
| territory | +10.0 s | 10.0 s | 0% |
| spread_y_diff | +7.2 s | 10.0 s | 0% |

## Per feature (oriented AUC vs proxy)

| feature | AUC | coverage | ball visible | ball missing |
|---|---|---|---|---|
| area_diff | 0.900 | 100% | 0.884 | 0.942 |
| spread_y_diff | 0.888 | 100% | 0.893 | 0.836 |
| area_h | 0.865 | 100% | 0.851 | 0.916 |
| press_diff | 0.854 | 100% | 0.824 | 0.928 |
| press_h | 0.823 | 100% | 0.794 | 0.891 |
| margin_10s | 0.807 | 94% | 0.833 | 0.636 |
| area_a | 0.805 | 100% | 0.799 | 0.844 |
| n_opp_half_diff | 0.797 | 100% | 0.744 | 0.931 |
| n_opp_half_h | 0.797 | 100% | 0.744 | 0.933 |
| spread_x_diff | 0.791 | 100% | 0.738 | 0.923 |
| n_opp_half_a | 0.787 | 100% | 0.734 | 0.925 |
| centroid_x_h | 0.784 | 100% | 0.722 | 0.929 |
| centroid_x_diff | 0.783 | 100% | 0.720 | 0.931 |
| territory | 0.783 | 100% | 0.720 | 0.931 |
| spread_y_h | 0.781 | 100% | 0.808 | 0.704 |
| centroid_x_a | 0.780 | 100% | 0.716 | 0.929 |
| near_share_10s | 0.778 | 94% | 0.817 | 0.582 |
| margin_3s | 0.771 | 86% | 0.781 | 0.604 |
| front_line_diff | 0.765 | 100% | 0.698 | 0.926 |
| spread_x_h | 0.762 | 100% | 0.706 | 0.918 |
| spread_x_a | 0.760 | 100% | 0.726 | 0.849 |
| front_line_h | 0.759 | 100% | 0.690 | 0.922 |
| front_line_a | 0.757 | 100% | 0.696 | 0.912 |
| length_diff | 0.755 | 100% | 0.735 | 0.822 |
| def_line_a | 0.745 | 100% | 0.668 | 0.933 |
| def_line_diff | 0.744 | 100% | 0.665 | 0.934 |
| line_gap | 0.744 | 100% | 0.665 | 0.934 |
| near_share_3s | 0.742 | 86% | 0.756 | 0.587 |
| def_line_h | 0.740 | 100% | 0.662 | 0.926 |
| margin_1s | 0.740 | 83% | 0.742 | 0.625 |
| press_a | 0.738 | 100% | 0.710 | 0.827 |
| margin_before_gap | 0.722 | 100% | 0.742 | 0.618 |
| margin | 0.722 | 81% | 0.722 | - |
| near_share_1s | 0.718 | 83% | 0.721 | 0.619 |
| mean_speed_diff | 0.715 | 100% | 0.759 | 0.505 |
| spread_y_a | 0.689 | 100% | 0.728 | 0.656 |
| ball_fwd_1s | 0.687 | 83% | 0.688 | 0.693 |
| d_a | 0.686 | 81% | 0.686 | - |
| ball_fwd_speed_home | 0.679 | 81% | 0.679 | - |
| n2_diff | 0.674 | 100% | 0.705 | 0.500 |
| dwell_diff | 0.671 | 100% | 0.703 | 0.500 |
| d_h | 0.665 | 81% | 0.665 | - |
| margin_after_gap | 0.662 | 100% | 0.742 | 0.640 |
| length_a | 0.659 | 100% | 0.667 | 0.654 |
| centroid_fwd_speed_h | 0.655 | 100% | 0.651 | 0.611 |
| centroid_fwd_speed_diff | 0.650 | 100% | 0.648 | 0.607 |
| centroid_fwd_speed_a | 0.642 | 100% | 0.643 | 0.586 |
| fwd_share_diff | 0.638 | 100% | 0.634 | 0.599 |
| fwd_share_a | 0.634 | 100% | 0.643 | 0.562 |
| nearest_fwd_speed_h | 0.620 | 81% | 0.620 | - |
| fwd_share_h | 0.620 | 100% | 0.607 | 0.603 |
| n2_h | 0.618 | 100% | 0.632 | 0.500 |
| length_h | 0.616 | 100% | 0.590 | 0.689 |
| nearest_fwd_speed_diff | 0.615 | 81% | 0.615 | - |
| mean_speed_h | 0.609 | 100% | 0.670 | 0.521 |
| dwell_h | 0.608 | 100% | 0.624 | 0.500 |
| nearest_fwd_speed_a | 0.606 | 81% | 0.606 | - |
| n5_diff | 0.605 | 100% | 0.620 | 0.500 |
| n2_a | 0.598 | 100% | 0.638 | 0.500 |
| ball_ahead_a | 0.597 | 81% | 0.597 | - |
| ball_x_home_frame | 0.594 | 81% | 0.594 | - |
| ball_ahead_diff | 0.593 | 81% | 0.593 | - |
| dwell_a | 0.592 | 100% | 0.628 | 0.500 |
| ball_ahead_h | 0.588 | 81% | 0.588 | - |
| n5_h | 0.584 | 100% | 0.569 | 0.500 |
| n5_a | 0.563 | 100% | 0.605 | 0.500 |
| closing_diff | 0.559 | 81% | 0.559 | - |
| mean_speed_all | 0.550 | 100% | 0.590 | 0.523 |
| ball_to_restart_spot | 0.543 | 81% | 0.543 | - |
| ball_frames_since_seen | 0.541 | 100% | 0.500 | 0.508 |
| ball_gap_len | 0.538 | 100% | 0.500 | 0.567 |
| relv_diff | 0.537 | 81% | 0.537 | - |
| ball_frames_until_seen | 0.537 | 100% | 0.500 | 0.577 |
| relv_h | 0.536 | 81% | 0.536 | - |
| n_players_h | 0.534 | 100% | 0.519 | 0.554 |
| relv_a | 0.527 | 81% | 0.527 | - |
| n_players_a | 0.526 | 100% | 0.514 | 0.540 |
| closing_a | 0.525 | 81% | 0.525 | - |
| last_seen_outside | 0.519 | 100% | 0.506 | 0.537 |
| closing_h | 0.518 | 81% | 0.518 | - |
| last_seen_to_line | 0.517 | 100% | 0.502 | 0.526 |
| mean_speed_a | 0.512 | 100% | 0.504 | 0.516 |
| ball_outside | 0.506 | 81% | 0.506 | - |
| ball_speed | 0.506 | 81% | 0.506 | - |
| ball_to_line | 0.502 | 81% | 0.502 | - |
| ball_still_frames | 0.501 | 100% | 0.503 | 0.500 |
| n_players_diff | 0.501 | 100% | 0.502 | 0.508 |

## Redundant pairs (|r| > 0.95)

- n_opp_half_diff ~ n_opp_half_h (r = 0.977)
- n_opp_half_diff ~ n_opp_half_a (r = -0.976)
- centroid_x_h ~ centroid_x_diff (r = 0.991)
- centroid_x_h ~ territory (r = 0.991)
- centroid_x_h ~ centroid_x_a (r = -0.965)
- centroid_x_h ~ front_line_diff (r = 0.964)
- centroid_x_h ~ def_line_diff (r = 0.974)
- centroid_x_h ~ line_gap (r = 0.974)
- centroid_x_h ~ def_line_h (r = 0.958)
- centroid_x_diff ~ territory (r = 1.0)
- centroid_x_diff ~ centroid_x_a (r = -0.992)
- centroid_x_diff ~ front_line_diff (r = 0.975)
- centroid_x_diff ~ def_line_a (r = -0.963)
- centroid_x_diff ~ def_line_diff (r = 0.983)
- centroid_x_diff ~ line_gap (r = 0.983)
- centroid_x_diff ~ def_line_h (r = 0.96)
- territory ~ centroid_x_a (r = -0.992)
- territory ~ front_line_diff (r = 0.975)
- territory ~ def_line_a (r = -0.963)
- territory ~ def_line_diff (r = 0.983)
- territory ~ line_gap (r = 0.983)
- territory ~ def_line_h (r = 0.96)
- centroid_x_a ~ front_line_diff (r = -0.969)
- centroid_x_a ~ def_line_a (r = 0.962)
- centroid_x_a ~ def_line_diff (r = -0.974)
- centroid_x_a ~ line_gap (r = -0.974)
- front_line_diff ~ front_line_a (r = -0.955)
- front_line_diff ~ def_line_diff (r = 0.952)
- front_line_diff ~ line_gap (r = 0.952)
- def_line_a ~ def_line_diff (r = -0.977)
- def_line_a ~ line_gap (r = -0.977)
- def_line_diff ~ line_gap (r = 1.0)
- def_line_diff ~ def_line_h (r = 0.98)
- line_gap ~ def_line_h (r = 0.98)
- margin_1s ~ margin_before_gap (r = 1.0)
- margin_1s ~ margin_after_gap (r = 1.0)
- margin_before_gap ~ margin_after_gap (r = 1.0)
- ball_fwd_1s ~ ball_fwd_speed_home (r = 0.989)
- centroid_fwd_speed_h ~ centroid_fwd_speed_diff (r = 0.988)
- centroid_fwd_speed_diff ~ centroid_fwd_speed_a (r = -0.987)
- ball_ahead_a ~ ball_ahead_diff (r = -0.99)
- ball_ahead_a ~ ball_ahead_h (r = -0.959)
- ball_ahead_diff ~ ball_ahead_h (r = 0.989)

## Dead-ball cues vs set-piece windows

```
{
 "mean_speed_all": {
  "auc": 0.5839833665625673,
  "coverage": 1.0
 },
 "ball_speed": {
  "auc": 0.5023471671134633,
  "coverage": 0.5921615037423594
 },
 "ball_still_frames": {
  "auc": 0.5038615742080563,
  "coverage": 1.0
 },
 "ball_outside": {
  "auc": 0.5235512654571084,
  "coverage": 0.5921615037423594
 },
 "ball_to_line": {
  "auc": 0.6000015190540884,
  "coverage": 0.5921615037423594
 },
 "ball_to_restart_spot": {
  "auc": 0.5028753379203966,
  "coverage": 0.5921615037423594
 },
 "ball_frames_since_seen": {
  "auc": 0.5721155233947457,
  "coverage": 1.0
 },
 "last_seen_to_line": {
  "auc": 0.567925086224828,
  "coverage": 1.0
 }
}
```

## Sanity

```
{
 "d_min": 0.0020249539520591497,
 "d_p99": 31.10407829284668,
 "mean_speed_all_p50_p99": [
  1.4712939262390137,
  4.763144016265869
 ],
 "n_players_h_max": 16,
 "ball_feature_coverage": 0.5921615037423594,
 "shape_feature_coverage": 0.9952976576417785
}
```