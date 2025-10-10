from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .utils import compute_flips, format_table, read_jsonl


def load_results(run_dir: Path):
    base_records = read_jsonl(run_dir / "first_turn.jsonl")
    followups_dir = run_dir / "followups"
    followup_records: Dict[str, List[dict]] = {}
    if followups_dir.exists():
        for path in sorted(followups_dir.glob("*.jsonl")):
            followup_records[path.stem] = read_jsonl(path)
    return base_records, followup_records


def summarize(base_records: List[dict], followup_records: Dict[str, List[dict]], only: List[str] | None = None) -> str:
    base_n = len(base_records)
    if base_n == 0:
        raise ValueError("No base records found in run directory")
    base_correct = sum(int(rec.get("correct", False)) for rec in base_records)
    base_acc = base_correct / base_n if base_n else 0.0

    headers = ["Followup", "n", "Accuracy", "Δ vs Base", "Flip C→I", "Flip I→C", "Flip Rate"]
    rows: List[List[str]] = [
        [
            "first_turn",
            str(base_n),
            f"{base_acc*100:.1f}%",
            "-",
            "-",
            "-",
            "-",
        ]
    ]

    for name, records in sorted(followup_records.items()):
        if only and name not in only:
            continue
        n = len(records)
        correct = sum(int(rec.get("correct", False)) for rec in records)
        acc = correct / n if n else 0.0
        flips = compute_flips(base_records, records)
        rows.append(
            [
                name,
                str(n),
                f"{acc*100:.1f}%",
                f"{(acc - base_acc)*100:+.1f}%",
                str(flips["flip_CI"]),
                str(flips["flip_IC"]),
                f"{flips["flip_rate"]*100:.1f}%",
            ]
        )

    return format_table(headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MedQA followup evaluation results")
    parser.add_argument("--run-dir", required=True, help="Path to a run directory produced by run.py")
    parser.add_argument("--followup", action="append", help="Restrict summary to specific followup names")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_records, followup_records = load_results(run_dir)
    table = summarize(base_records, followup_records, args.followup)
    print("Evaluation Summary:\n" + table)


if __name__ == "__main__":
    main()
