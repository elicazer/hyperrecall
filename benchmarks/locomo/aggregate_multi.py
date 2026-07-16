#!/usr/bin/env python3
"""Aggregate all completed unified-harness judgments into Markdown tables."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "multi"
CATEGORIES = {1: "c1", 2: "c2", 3: "c3", 4: "c4", 5: "c5"}
METRICS = ("strict", "lax", "f1", "bleu1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(rows: list[dict[str, Any]], metric: str) -> float:
    if not rows:
        return 0.0
    if metric == "strict":
        return sum(row["judge_label"] == "correct" for row in rows) / len(rows)
    if metric == "lax":
        return sum(row["judge_label"] in {"correct", "partial"} for row in rows) / len(rows)
    return sum(float(row[metric]) for row in rows) / len(rows)


def system_cost(directory: Path) -> tuple[float, bool]:
    total, estimated = 0.0, False
    for pattern in ("*.system_cost.json", "*.answer_cost.json", "*.judge_cost.json"):
        for path in directory.glob(pattern):
            data = json.loads(path.read_text())
            total += float(data.get("usd") or 0)
            estimated = estimated or bool(data.get("estimated"))
    return total, estimated


def main() -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = sorted(RUN_ROOT.glob("*/*.judged.jsonl"))
    if not files:
        print(f"No judged outputs found under {RUN_ROOT}")
        return 1
    for path in files:
        grouped[path.parent.name].extend(read_jsonl(path))

    summaries: dict[str, Any] = {}
    for system, rows in sorted(grouped.items()):
        by_cat = {cat: [row for row in rows if int(row["category"]) == cat] for cat in CATEGORIES}
        cost, estimated = system_cost(RUN_ROOT / system)
        summaries[system] = {
            "n": len(rows), "metrics": {
                metric: {"overall": score(rows, metric), **{
                    CATEGORIES[cat]: score(cat_rows, metric) for cat, cat_rows in by_cat.items()
                }} for metric in METRICS
            },
            "labels": dict(Counter(row["judge_label"] for row in rows)),
            "cost_usd": cost, "cost_estimated": estimated,
        }

    lines = []
    for metric in METRICS:
        lines += [f"## {metric}", "", "| system | overall | c1 | c2 | c3 | c4 | c5 | cost |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for system, summary in summaries.items():
            values = summary["metrics"][metric]
            cost = f"${summary['cost_usd']:.4f}" + (" est." if summary["cost_estimated"] else "")
            lines.append("| " + " | ".join([
                system, *(f"{values[key]:.3f}" for key in ("overall", "c1", "c2", "c3", "c4", "c5")), cost
            ]) + " |")
        lines.append("")
    output = "\n".join(lines)
    print(output)
    (RUN_ROOT / "summary.md").write_text(output + "\n")
    (RUN_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
