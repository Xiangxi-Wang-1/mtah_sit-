# Layer 3 evaluation

| check | value | target | |
|---|---|---|---|
| 2nd-half pre-kickoff is dead | 0.993 | > 0.95 | PASS |
| minute before each after-goal kick-off is dead | 1.0 | > 0.85 | PASS |
| ball > 1.5 m beyond a line is dead | 0.974 | > 0.97 | PASS |
| set-piece windows contain a restart | 0.948 | >= 0.85 | PASS |
| effective playing time (min; LaLiga avg ~53.4) | 50.4 | 45-60 | PASS |
| number of stoppages | 122 | 90-150 | PASS |
| stoppages shorter than 3 s (implausible) | 1 | <= 10 | PASS |
| players slower when dead (median m/s, dead vs alive) | [1.02, 2.2] | dead < 0.75 x alive | PASS |

```
{
 "dead_share": 0.5240499316995789,
 "dead_minutes": 55.49933333333333,
 "alive_minutes": 50.40533333333333,
 "stoppages": 122,
 "stoppage_s_p25_p50_p75": [
  6.49,
  16.5,
  35.33
 ],
 "stoppages_under_3s": 1,
 "alive_spells_under_3s": 4,
 "second_half_prekickoff_dead": 0.9931833674164963,
 "before_goal_kickoffs_dead": 1.0,
 "set_piece_restarts_found": 0.9482758620689655,
 "set_piece_restarts_by_type": {
  "Kick Off (After Goal)": [
   4,
   4
  ],
  "Kick Off (Start)": [
   1,
   1
  ],
  "Corner Kick": [
   11,
   12
  ],
  "Throw-in": [
   26,
   26
  ],
  "Goal Kick": [
   9,
   10
  ],
  "Free Kick": [
   4,
   5
  ]
 },
 "ball_far_out_dead": 0.9743472285845167,
 "player_speed_dead_vs_alive": [
  1.0244851112365723,
  2.196250081062317
 ]
}
```