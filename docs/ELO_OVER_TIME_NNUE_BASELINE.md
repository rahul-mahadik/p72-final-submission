# Elo Over Time: Alpha-Beta vs NNUE-Lite

This report treats NNUE-lite as Elo 1000 and reports alpha-beta's relative Elo at increasing per-move search-budget checkpoints.

| Checkpoint | Move Budget (ms) | Games | Score vs NNUE | Elo Delta | Estimated Elo | Illegal | Crashes | Avg Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 4 | 0.5000 | -0.0 | 1000.0 | 0 | 0 | 18.39 |
| 2 | 50 | 4 | 0.3750 | -88.7 | 911.3 | 0 | 0 | 50.10 |
| 3 | 100 | 4 | 0.3750 | -88.7 | 911.3 | 0 | 0 | 98.05 |

## Notes

- Runtime JSON and PGN artifacts are generated under `artifacts/` and ignored by git.
- Re-run with `python3 scripts/elo_over_time.py --games-per-position 4` for a less noisy curve.
