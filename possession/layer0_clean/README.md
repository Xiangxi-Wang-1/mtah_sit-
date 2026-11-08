# Layer 0 · Clean the data

Turns the raw tactical ATD into clean arrays in metres, indexed by video frame (= submission row).

| file | does |
|---|---|
| `parse.py` | reads the raw ATD and metadata into numpy, vectorised (~3 s) |
| `ball.py` | drops over-speed glides and isolated spikes, fills gaps of ≤ 1 s with a cubic Hermite curve, smooths, differentiates, records gap context |
| `players.py` | splits slot runs at impossible jumps, removes same-team duplicates, smooths, differentiates |
| `teams.py` | repairs cross-team identity switches using the 11-a-side count |
| `direction.py` | attack direction per team per half |
| `run.py` | runs it all, writes `cache/layer0_tactical.npz` (array list in its docstring) |
| `evaluate.py` | self-supervised checks, writes `reports/layer0/report.md` and figures |

## What the data turned out to be

- **The ball track is already smoothed by the provider**, so there is little jitter to remove. When
  the tracker re-locks onto another object it glides there at 45–75 m/s; those frames are removed.
- **Missing ball comes in long blocks**, not short gaps: 132 of 150 gaps are over 5 s. Short-gap
  filling touches only a few dozen frames. Long gaps start on average from a slow ball and end on a
  fast one (a restart), which layer 3 (dead ball) uses; their edges are exported as gap context.
- **The ATD ball speed column is not a ball speed**: present on 8% of ball frames, uncorrelated with
  the ball's motion. Not used. The player speed column is fine (correlation 0.96 with ours).
- **Track slots switch between teams.** One team showed 12+ against 10 or fewer on 10.5% of graded
  frames. The repair brings that to 3.1%; the rest has no visible switch point and is flagged
  `team_count_anomaly` so later layers can down-weight team labels there.

## How it is evaluated

No ground truth is used. Physics (nothing faster than possible), agreement with the ATD's own
player speeds, provider metadata (attack directions), and two injected-error benchmarks:
hide real ball samples and refill them; inject realistic identity switches and wrong-team track
births into clean stretches of the real tracking and see whether the repair finds them.

## Known limits

- Within ~0.5 m, a slot switching onto an opponent and an opponent re-acquired under the wrong team
  leave identical position data, so 6–8% of injected errors get the wrong slot relabelled.
  Fixing that needs shirt colour from the video.
- Referees and staff near the touchline are not separated from players.
