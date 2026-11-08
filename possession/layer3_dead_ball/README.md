# Layer 3 · Dead-ball detector

    python -m possession.layer3_dead_ball.run              # ~1 s -> cache/layer3_tactical.npz
    python -m possession.layer3_dead_ball.evaluate --sweep # self-supervised checks + weight sweep

## How it works

1. **Evidence per frame** (`deadball.evidence`), in log-odds of "dead", from cues layers 0–2 already
   computed: ball beyond a line; ball missing (more if for > 5 s and last seen at a line); everyone
   slowing down (median player speed halves at a stoppage); ball at rest, more on a restart spot;
   a kick-off formation (each team wholly in its own half); the first half of a Smart Tagging
   set-piece window. Play-is-on cues count against: a fast ball inside the pitch, a player
   controlling the ball away from the lines.
2. **Two-state Viterbi** (alive / dead) with a switch cost, so one slow moment does not stop play.
3. **Boundaries snapped to physical events**: a restart to the kick (ball accelerating from rest,
   or re-appearing already moving), a stoppage to the ball crossing a line or being lost.
4. **Rules of the game**: play cannot restart while the ball is beyond a line (a throw-in starts
   outside; play resumes when it comes in); a kick-off formation that breaks up is the kick-off,
   even if the ball is not seen; everything in a half before its kick-off is dead; no stoppage
   shorter than 3 s and no spell of play shorter than 2 s between two stoppages.

## How it is evaluated (no ground truth)

| check | result | target |
|---|---|---|
| 2nd half graded from 110,808 but kicks off at 113,742 (FAQ) - dead | 99.3% | > 95% |
| minute before each kick-off after a goal - dead | 100% | > 85% |
| ball > 1.5 m beyond a line - dead | 97.4% | > 97% |
| Smart Tagging set pieces with a restart in their window | 56 / 59 | ≥ 85% |
| effective playing time | 50.4 min | 45–60 (LaLiga avg ~53.4) |
| stoppages | 122 | 90–150 |
| stoppages shorter than 3 s | 1 | ≤ 10 |
| median player speed, dead vs alive | 1.0 vs 2.2 m/s | dead < 0.75 × alive |

All throw-ins (26/26) and kick-offs (5/5) are found. The three missed set pieces (a corner, a goal
kick, a free kick) are ones where the tracker never sees the ball, so the restart has no visible
moment. Smart Tagging windows are 35 s long and cover set-up, kick and aftermath (the restart sits
4–15 s in), so they confirm a restart happened but cannot time it - timing comes from the kick.

Weight sweeps (`--sweep`) sit on a plateau: every weight can be halved or doubled without changing
which checks pass, so the defaults are physical choices, not tuned values.

## Effect on the read-out

See `possession/compare_models.py` / `reports/model_comparison.md`. Adding layer 3 to the layer 0–2
read-out takes spellF1 on the public 5 minutes from 0.286 to 0.425 (baseline 0.326) and dead-ball
frames correctly called from 34% to 86%. Nothing in layer 3 was chosen from the public window.
