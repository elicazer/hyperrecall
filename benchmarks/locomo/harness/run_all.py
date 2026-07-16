#!/usr/bin/env python3
"""Run MeshMind, vector RAG, Mem0, and Zep under one LoCoMo protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCOMO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(LOCOMO_ROOT))

from harness.judge import GeminiJudge
from harness.load import Conversation, load
from harness.systems.base import System

SYSTEMS = ("meshmind", "vector_rag", "mem0", "zep")
CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}
HEARTBEAT = Path("/tmp/david-worker-hb/C.hb")


def heartbeat() -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n"))


def load_gemini_env() -> None:
    path = Path.home() / ".config" / "openclaw" / "gemini.env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("export "):
                key, _, value = line[7:].partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # The benchmark contract is Gemini Developer API via GEMINI_API_KEY only.
    os.environ.pop("GOOGLE_API_KEY", None)


def make_system(name: str, conv_id: str) -> System:
    if name == "meshmind":
        from harness.systems.meshmind_system import MeshMindSystem
        return MeshMindSystem(LOCOMO_ROOT, conv_id)
    if name == "vector_rag":
        from harness.systems.vector_rag_system import VectorRagSystem
        return VectorRagSystem(LOCOMO_ROOT, conv_id)
    if name == "mem0":
        from harness.systems.mem0_system import Mem0System
        return Mem0System(LOCOMO_ROOT, conv_id)
    if name == "zep":
        from harness.systems.zep_system import ZepSystem
        return ZepSystem(LOCOMO_ROOT, conv_id)
    raise ValueError(f"unknown system: {name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_score(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        labels = Counter(row["judge"]["label"] for row in items)
        return {
            "n": n,
            "strict": labels["correct"] / n if n else 0.0,
            "lax": (labels["correct"] + labels["partial"]) / n if n else 0.0,
            "f1": sum(row["judge"]["f1"] for row in items) / n if n else 0.0,
            "bleu1": sum(row["judge"]["bleu1"] for row in items) / n if n else 0.0,
            "labels": dict(labels),
        }

    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[int(row["category"])].append(row)
    return {
        "overall": group_score(rows),
        "categories": {
            str(cat): {"name": CATEGORY_NAMES[cat], **group_score(by_category[cat])}
            for cat in CATEGORY_NAMES
        },
    }


def run_system(name: str, conv: Conversation, run_dir: Path, evaluator: GeminiJudge) -> dict[str, Any]:
    output = run_dir / f"{name}.jsonl"
    completed = {int(row["idx"]) for row in read_jsonl(output)}
    system = make_system(name, conv.sample_id)
    started = time.time()
    try:
        system.ingest(conv)
        for idx, qa in enumerate(conv.qa):
            if idx in completed:
                continue
            heartbeat()
            retrieval = system.retrieve(qa.question)
            prediction = evaluator.answer(retrieval.context, qa.question)
            judgment = evaluator.score(qa.question, qa.answer, prediction)
            append_jsonl(output, {
                "idx": idx, "system": name, "conv": conv.sample_id,
                "question": qa.question, "gold": qa.answer,
                "category": qa.category, "evidence": qa.evidence,
                "context": retrieval.context, "retrieval": retrieval.metadata,
                "prediction": prediction, "judge": judgment,
            })
            if (idx + 1) % 10 == 0:
                print(f"[{name}] {idx + 1}/{len(conv.qa)}", flush=True)
    finally:
        cost = system.cost_record()
        system.close()
    rows = read_jsonl(output)
    return {
        **summarize(rows), "complete": len(rows) == len(conv.qa),
        "n_expected": len(conv.qa), "cost": cost, "seconds": time.time() - started,
    }


def main() -> int:
    load_gemini_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv", default="26", help="LoCoMo conversation number (default: 26)")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--run-id", help="output directory name; defaults to a UTC timestamp")
    args = parser.parse_args()
    requested = [item.strip() for item in args.systems.split(",") if item.strip()]
    unknown = [item for item in requested if item not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems: {', '.join(unknown)}")
    conv_id = args.conv if args.conv.startswith("conv-") else f"conv-{args.conv}"
    try:
        conv = next(item for item in load() if item.sample_id == conv_id)
    except StopIteration:
        parser.error(f"conversation not found: {conv_id}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = LOCOMO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat()
    evaluator = GeminiJudge()
    summary: dict[str, Any] = {
        "schema_version": 1, "conversation": conv_id,
        "model": "gemini-2.5-pro", "systems": {}, "skipped": {},
    }
    for name in requested:
        print(f"[{name}] starting", flush=True)
        try:
            summary["systems"][name] = run_system(name, conv, run_dir, evaluator)
        except RuntimeError as exc:
            reason = str(exc)
            summary["skipped"][name] = reason
            print(f"[{name}] SKIPPED: {reason}", flush=True)
        except Exception as exc:
            partial = read_jsonl(run_dir / f"{name}.jsonl")
            if partial:
                summary["systems"][name] = {
                    **summarize(partial), "complete": False,
                    "n_expected": len(conv.qa), "cost": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            summary["skipped"][name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    summary["model_cost"] = evaluator.cost_record()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {run_dir / 'summary.json'}")
    return 0 if summary["systems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
