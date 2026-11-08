# Layer 5 evaluation

Evaluate layer 5 without any ground truth (and the public window as a check only).

    python -m possession.layer5_sequence_decoder.evaluate          # current settings
    python -m possession.layer5_sequence_decoder.evaluate --sweep  # sweep the main settings

Selection uses only:
  - agreement with the Smart Tagging proxy (all live frames, and stale ones)
  - Smart Tagging turnovers found within 10 s, and their lag
  - consistency: certain-possession frames (layer 4 labels) decoded to the same team
  - reception consistency: share of uncontested receptions (layer 2) whose team holds the ball
    while he controls it. The challenge counts any controlled possession, however brief: the
    public window (CHALLENGE.md: "use it to understand how these cases are labelled") has
    possessions of 0.8 s and 1.3 s - a reception by the other team is a possession.
  - kick consistency: share of kicks (layer 2: a player plays the ball away first time) whose
    team has the ball right after - the team that last played the ball keeps it until the other
    team controls it, which is how the public window labels clearances and first-time passes
  - possessions per minute of play (sanity: not below 2)
The public 5 minutes is printed next to them but never used to choose settings.

```
{
 "current": {
  "proxy_agreement": 0.8916208852561526,
  "proxy_agreement_stale": 0.9444117647058824,
  "turnovers_found": 1.0,
  "turnover_abs_lag_s": 5.2,
  "certain_consistency": 0.9989708530892785,
  "receptions_honoured": 0.9917763157894737,
  "kicks_honoured": 0.9310344827586207,
  "possessions_per_min": 2.6438872696827964,
  "possession_s_median": 7.48,
  "short_possessions_share": 0.125,
  "public_spellF1_check_only": 0.5454545454545455,
  "public_macroF1_check_only": 0.8298581159669703,
  "public_possessions_check_only": 16,
  "objective": 5.777384735035941
 }
}
```
