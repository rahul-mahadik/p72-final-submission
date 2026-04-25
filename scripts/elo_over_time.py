#!/usr/bin/env python3
"""Generate an Elo-over-time report for one engine against a baseline.

The default report compares alpha-beta against NNUE-lite. "Time" is represented as increasing per-move
search budgets, which gives reviewers a reproducible way to see how relative
strength changes as the engine receives more compute.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tournament.runner import evaluate_engine


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the report generator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="alphabeta")
    parser.add_argument("--baseline", default="nnue_lite")
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--games-per-position", type=int, default=2)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/elo_over_time")
    parser.add_argument("--report-path", default="docs/ELO_OVER_TIME_NNUE_BASELINE.md")
    parser.add_argument("--csv-path", default="docs/elo_over_time_nnue_baseline.csv")
    return parser.parse_args()


def build_rows(args: argparse.Namespace) -> list[dict]:
    """Run evaluations for each search-budget checkpoint."""
    rows: list[dict] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, budget in enumerate(args.budgets, start=1):
        summary = evaluate_engine(
            args.candidate,
            baseline_kind=args.baseline,
            heldout=args.heldout,
            move_budget_ms=budget,
            games_per_position=args.games_per_position,
            output_dir=output_dir,
        )
        rows.append(
            {
                "checkpoint": index,
                "move_budget_ms": budget,
                "candidate": args.candidate,
                "baseline": args.baseline,
                "games": summary["games"],
                "score_vs_baseline": round(summary["win_rate"], 4),
                "elo_delta_vs_baseline": round(summary["elo_delta"], 1),
                "estimated_candidate_elo": round(summary["estimated_elo"], 1),
                "reference_baseline_elo": summary["reference_elo"],
                "illegal_moves": summary["illegal_moves"],
                "crashes": summary["crashes"],
                "avg_move_latency_ms": round(summary["avg_move_latency_ms"], 2),
                "artifact_json": summary["artifact_files"][0],
                "artifact_pgn": summary["artifact_files"][1],
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write checkpoint metrics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write a compact Markdown report for reviewers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Elo Over Time: Alpha-Beta vs NNUE-Lite\n\n"
        "This report treats NNUE-lite as Elo 1000 and "
        "reports alpha-beta's relative Elo at increasing "
        "per-move search-budget checkpoints.\n\n"
    )
    columns = [
        "Checkpoint",
        "Move Budget (ms)",
        "Games",
        "Score vs NNUE",
        "Elo Delta",
        "Estimated Elo",
        "Illegal",
        "Crashes",
        "Avg Latency (ms)",
    ]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row["checkpoint"]),
                    str(row["move_budget_ms"]),
                    str(row["games"]),
                    f"{row['score_vs_baseline']:.4f}",
                    f"{row['elo_delta_vs_baseline']:.1f}",
                    f"{row['estimated_candidate_elo']:.1f}",
                    str(row["illegal_moves"]),
                    str(row["crashes"]),
                    f"{row['avg_move_latency_ms']:.2f}",
                ]
            )
            + " |"
        )
    notes = (
        "\n\n## Notes\n\n"
        "- Runtime JSON and PGN artifacts are generated under `artifacts/` and ignored by git.\n"
        "- Re-run with `python3 scripts/elo_over_time.py --games-per-position 4` for a less noisy curve.\n"
    )
    path.write_text(header + "\n".join(table) + notes, encoding="utf-8")


def main() -> None:
    """Generate CSV, Markdown, and print JSON rows."""
    args = parse_args()
    rows = build_rows(args)
    write_csv(rows, Path(args.csv_path))
    write_markdown(rows, Path(args.report_path))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
