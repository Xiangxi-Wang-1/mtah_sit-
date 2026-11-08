# Layer 0 evaluation

| check | value | target | |
|---|---|---|---|
| one tracking record per video frame (199,319) | 199319 | all 199,319 | PASS |
| raw ball coverage matches DATA.md (59.3%) | 59.3 | 59.3 +/- 0.5 % | PASS |
| false detections removed | 0.13 | < 3 % of graded frames | PASS |
| ball steps faster than V_MAX after cleaning | 0.03 | < 0.1 % of steps | PASS |
| ball smoothing moves a sample (p95) | 0.0 | < 0.30 m | PASS |
| 12-vs-10 frames: before -> after repair (%) | 10.5 -> 3.1 | cut by > 65 % | PASS |
| every frame still over 11 is flagged team_count_anomaly | 5.23 | all flagged | PASS |
| player speeds over 11 m/s after cleaning | 0.044 | < 0.1 % | PASS |
| player speed agrees with ATD speed column (corr) | 0.956 | > 0.90 | PASS |
| attack directions flip at half time and match the goalkeepers | {'1': (1, -1), '2': (-1, 1)} | consistent + flipped | PASS |
| first-half directions agree with the provider metadata | {'home': 'left_to_right', 'away': 'right_to_left'} | agree | PASS |
| team repair recovers injected mid-run identity switches (recall, of 97 that break the count)% | 0.911 | >= 0.90 | PASS |
| team repair relabels a wrong slot (swap) | 0.08 | <= 0.10 | PASS |
| team repair recovers injected tracks born with the wrong team (recall, of 97 that break the count)% | 0.932 | >= 0.90 | PASS |
| team repair relabels a wrong slot (birth) | 0.06 | <= 0.10 | PASS |
| gap fill p90 max-error at the longest filled gap (25 frames) | 0.33 | < 1.5 m (control radius) | PASS |

```
{
 "coverage": {
  "observed": 0.5917838055609762,
  "filled": 0.00037769818138325666,
  "missing": 0.40661727213783466,
  "removed": 0.0012904687863927935,
  "raw_known": 0.593074274347369,
  "clean_known": 0.5921615037423594
 },
 "gaps": {
  "1-1": [
   2,
   1
  ],
  "2-5": [
   12,
   7
  ],
  "6-12": [
   2,
   4
  ],
  "13-25": [
   1,
   7
  ],
  "26-125": [
   1,
   1
  ],
  "126-inf": [
   132,
   132
  ]
 },
 "short_gaps_unfilled": [
  19,
  17
 ],
 "ball_implied_speed": {
  "raw_p99": 24.603660583496094,
  "clean_p99": 23.84192657470703,
  "raw_frac_over_vmax": 0.0020198798703024504,
  "clean_frac_over_vmax": 0.00029813241338188634
 },
 "ball_smoothing_residual_m": {
  "median": 0.00021477795962709934,
  "p95": 0.00045564878382720053
 },
 "atd_ball_speed_column": {
  "present_frac_of_ball_frames": 0.07830046489905959,
  "corr_with_motion": 0.05772350002165896
 },
 "long_gaps": {
  "n": 133,
  "start_within_3m_of_line": 0.3458646616541353
 },
 "players_per_team_before_repair": {
  "frac_over_11": 0.06242091944327288,
  "frac_12_vs_10": 0.10537149763623888
 },
 "team_repairs": {
  "n": 66,
  "frames": 26479,
  "by_evidence": {
   "jump": 3,
   "contact+motion": 0,
   "start+reacquire": 0,
   "contact": 22,
   "start": 26
  }
 },
 "players_per_team": {
  "mean": [
   10.12,
   10.32
  ],
  "frac_9_to_11": 0.8861869480098453,
  "frac_over_11": 0.026250023606136336,
  "frac_under_8": 0.06306300635162441,
  "frac_12_vs_10": 0.03129858929729253
 },
 "player_duplicate_rate": 0.002710035427107114,
 "player_speed": {
  "p99": 6.556765556335449,
  "frac_over_11": 0.0004393992617106374
 },
 "player_speed_vs_atd": {
  "corr": 0.9563207974507908,
  "median_abs_diff": 0.10872691869735718
 },
 "direction": {
  "1": {
   "home_median_x": 50.696607506493436,
   "home": 1,
   "home_gk_median_x": 10.998165623376607,
   "away_median_x": 60.9457208961038,
   "away": -1,
   "away_gk_median_x": 97.13591322077907,
   "consistent": true
  },
  "2": {
   "home_median_x": 54.72223562337654,
   "home": -1,
   "home_gk_median_x": 92.68608109090896,
   "away_median_x": 44.596498733766175,
   "away": 1,
   "away_gk_median_x": 7.618197448051936,
   "consistent": true
  }
 },
 "ball_outside_pitch_frac": 0.026703802527931624,
 "team_repair_benchmark_swap": {
  "n": 150,
  "detectable_frac": 0.9733333333333334,
  "recall": 0.910958904109589,
  "wrong_relabel_rate": 0.08,
  "false_repair_when_undetectable": 0.0,
  "switch_point_abs_err_median": 0.0
 },
 "team_repair_benchmark_birth": {
  "n": 150,
  "detectable_frac": 0.9733333333333334,
  "recall": 0.9315068493150684,
  "wrong_relabel_rate": 0.06,
  "false_repair_when_undetectable": 0.0,
  "switch_point_abs_err_median": 0.0
 },
 "gap_fill_benchmark": {
  "1": {
   "linear": [
    0.004213481639698664,
    0.017768778388683917,
    439
   ],
   "hermite": [
    0.004150936789637189,
    0.018230194468677453,
    439
   ]
  },
  "2": {
   "linear": [
    0.010150102764587837,
    0.037773828081503,
    434
   ],
   "hermite": [
    0.007013941365564354,
    0.02474239222362761,
    434
   ]
  },
  "3": {
   "linear": [
    0.020032138319404435,
    0.06887220748399678,
    449
   ],
   "hermite": [
    0.009689035680069095,
    0.032973321219894,
    449
   ]
  },
  "5": {
   "linear": [
    0.04067048080773564,
    0.16507993957572017,
    440
   ],
   "hermite": [
    0.01387305878999698,
    0.05090749631219702,
    440
   ]
  },
  "8": {
   "linear": [
    0.10705540247283307,
    0.3729788923038112,
    452
   ],
   "hermite": [
    0.023437338934896162,
    0.07011456672948146,
    452
   ]
  },
  "10": {
   "linear": [
    0.14786710350258403,
    0.5255075541918904,
    422
   ],
   "hermite": [
    0.02480324976238534,
    0.07238275090674674,
    422
   ]
  },
  "12": {
   "linear": [
    0.19889527718303263,
    0.6528969860399992,
    430
   ],
   "hermite": [
    0.026180134244375257,
    0.06930010097233288,
    430
   ]
  },
  "15": {
   "linear": [
    0.27614610777410464,
    0.9187300650865935,
    424
   ],
   "hermite": [
    0.031426337917006664,
    0.08139242596748551,
    424
   ]
  },
  "20": {
   "linear": [
    0.4540329493820541,
    1.5328661483232457,
    376
   ],
   "hermite": [
    0.05428359261061748,
    0.15315798357849197,
    376
   ]
  },
  "25": {
   "linear": [
    0.7632719130753519,
    2.2381274042975567,
    389
   ],
   "hermite": [
    0.11481669484847133,
    0.33217210386523877,
    389
   ]
  }
 }
}
```