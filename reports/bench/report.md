# Bench score

Run the pipeline on the bench games and score it with the official scorer.

    python -m possession.bench.score                 # run layers 1-5 on bench1-3, then score
    python -m possession.bench.score --no-run        # score what is in the cache
    python -m possession.bench.score --label v1      # also append the result to reports/bench/history.md

Per game: spellF1 and macroF1 (official evaluation/possession_metrics.py), alive/dead F1, team
accuracy on alive frames, possession count. Two diagnostic swaps say which half of the problem
costs the most:
  our team + true dead    the pipeline's team calls, with the answer key's stoppages
  true team + our dead    the answer key's team calls, with the pipeline's stoppages

Nothing here is tuned on the challenge match; this is the score the pipeline is chosen on.

| | baseline | pipeline | our team + true dead | true team + our dead |
|---|---|---|---|---|
| spellF1 (mean of 3) | 0.361 | 0.681 | 0.699 | 0.882 |
| macroF1 (mean of 3) | 0.816 | 0.901 | 0.934 | 0.932 |
| alive_dead_F1 (mean of 3) | 0.938 | 0.940 | 1.000 | 0.940 |
| team_acc_alive (mean of 3) | 0.837 | 0.952 | 0.902 | 1.000 |
| spellF1 bench1 | 0.375 | 0.687 | 0.702 | 0.881 |
| possessions bench1 (truth 389) | 500 | 287 | 320 | 359 |
| spellF1 bench2 | 0.368 | 0.693 | 0.708 | 0.880 |
| possessions bench2 (truth 366) | 508 | 276 | 309 | 349 |
| spellF1 bench3 | 0.339 | 0.663 | 0.689 | 0.886 |
| possessions bench3 (truth 411) | 640 | 323 | 357 | 377 |

```
{
 "bench1": {
  "baseline": {
   "spellF1": 0.3752530933633296,
   "macroF1": 0.7954281722429949,
   "alive_dead_F1": 0.9225927362732831,
   "team_acc_alive": 0.8265463245919503,
   "possessions": 500,
   "truth_possessions": 389
  },
  "pipeline": {
   "spellF1": 0.68698224852071,
   "macroF1": 0.8914016149373637,
   "alive_dead_F1": 0.9296065217119495,
   "team_acc_alive": 0.9480019906118441,
   "possessions": 287,
   "truth_possessions": 389
  },
  "our team + true dead": {
   "spellF1": 0.701833568406206,
   "macroF1": 0.9127572384387573,
   "alive_dead_F1": 1.0,
   "team_acc_alive": 0.8691372797447268,
   "possessions": 320,
   "truth_possessions": 389
  },
  "true team + our dead": {
   "spellF1": 0.8807486631016042,
   "macroF1": 0.9236581820378548,
   "alive_dead_F1": 0.9296065217119495,
   "team_acc_alive": 1.0,
   "possessions": 359,
   "truth_possessions": 389
  }
 },
 "bench2": {
  "baseline": {
   "spellF1": 0.36796338672768886,
   "macroF1": 0.8378502025457119,
   "alive_dead_F1": 0.9419546864036578,
   "team_acc_alive": 0.856343037016568,
   "possessions": 508,
   "truth_possessions": 366
  },
  "pipeline": {
   "spellF1": 0.6928348909657321,
   "macroF1": 0.9052910420788325,
   "alive_dead_F1": 0.9384744596070903,
   "team_acc_alive": 0.9576486737454075,
   "possessions": 276,
   "truth_possessions": 366
  },
  "our team + true dead": {
   "spellF1": 0.7075555555555556,
   "macroF1": 0.9476339123857741,
   "alive_dead_F1": 1.0,
   "team_acc_alive": 0.9215443737240303,
   "possessions": 309,
   "truth_possessions": 366
  },
  "true team + our dead": {
   "spellF1": 0.8800000000000001,
   "macroF1": 0.9318736645328572,
   "alive_dead_F1": 0.9384744596070903,
   "team_acc_alive": 1.0,
   "possessions": 349,
   "truth_possessions": 366
  }
 },
 "bench3": {
  "baseline": {
   "spellF1": 0.33910561370123693,
   "macroF1": 0.815988242245575,
   "alive_dead_F1": 0.9498408069024183,
   "team_acc_alive": 0.8290226502768595,
   "possessions": 640,
   "truth_possessions": 411
  },
  "pipeline": {
   "spellF1": 0.6626702997275205,
   "macroF1": 0.9064952729341291,
   "alive_dead_F1": 0.9531074498948657,
   "team_acc_alive": 0.9489851769936217,
   "possessions": 323,
   "truth_possessions": 411
  },
  "our team + true dead": {
   "spellF1": 0.6885416666666666,
   "macroF1": 0.9428019501069022,
   "alive_dead_F1": 1.0,
   "team_acc_alive": 0.9142851316768793,
   "possessions": 357,
   "truth_possessions": 411
  },
  "true team + our dead": {
   "spellF1": 0.8857868020304569,
   "macroF1": 0.9389902374504141,
   "alive_dead_F1": 0.9531074498948657,
   "team_acc_alive": 1.0,
   "possessions": 377,
   "truth_possessions": 411
  }
 }
}
```
