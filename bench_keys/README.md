# Bench answer keys (Metrica's 3 open sample games)

Frame-level `frame,label` CSVs (H / A / D / X) in the challenge's format, for Metrica Sports'
open sample games (https://github.com/metrica-sports/sample-data). Nothing here comes from the
challenge match.

| file | what it is |
|---|---|
| `bench{g}_gt.csv` | answer key v2: Metrica's event log read with Metrica's own event definitions (`possession/bench/labels.py`) |
| `bench{g}_gt_v1.csv` | answer key v1 (first version, kept for comparison) |
| `bench{g}_baseline.csv` | the official baseline (nearest player, +/- 2 s vote) on the degraded bench tracking |

Regenerate with `python -m possession.bench.build` (see `possession/bench/README.md`).
