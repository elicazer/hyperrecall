#!/usr/bin/env python3
"""Unified, resumable LoCoMo runner for multiple memory systems."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(ROOT))

from harness.load import Conversation, load
from harness.metrics import bleu1, gold_string, token_f1
from harness.providers import ModelClient, PRICES, estimate_tokens

RUN_ROOT = ROOT / "runs" / "multi"
SUPPORTED_SYSTEMS = ("meshmind", "mem0", "zep", "vector_rag")
ANSWERERS = ("openai:gpt-4o", "gemini:gemini-2.5-flash", "bedrock:opus-4-8")
JUDGES = ("openai:gpt-4o", "gemini:gemini-2.5-flash")

# This exact template is applied after retrieval for every system. Adapters are
# prohibited from answering, rewriting the question, or changing this prompt.
ANSWER_PROMPT = """You are answering questions about a long-term conversation between two people. Use ONLY the provided memory context. If the context does not contain enough information to answer, reply exactly: 'I don't know.' Keep answers concise (typically 1-15 words).

Memory context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading an answer to a question about a long-term conversation.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Grade the prediction against the gold using ONE label:
- correct: matches the gold in meaning (paraphrases are acceptable).
- partial: on-topic and has the right entity/concept, but misses part of the gold.
- wrong: contradicts the gold, is unrelated, or does not answer.

If the gold is a date and a relative expression resolves to the same event, use partial. If the gold expects an unanswerable/refusal response and the prediction refuses, use correct.
Return only JSON: {{"label":"correct|partial|wrong","reason":"one short sentence"}}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}


def load_env() -> None:
    path = Path.home() / ".config" / "openclaw" / "gemini.env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("export "):
                key, _, value = line[7:].partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_conversations(spec: str, conversations: list[Conversation]) -> list[Conversation]:
    if spec == "all":
        return conversations
    requested = parse_csv(spec)
    by_id = {conv.sample_id: conv for conv in conversations}
    missing = [conv_id for conv_id in requested if conv_id not in by_id]
    if missing:
        raise ValueError(f"unknown conversation(s): {', '.join(missing)}")
    return [by_id[conv_id] for conv_id in requested]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def usage_delta(model: ModelClient, before: dict[str, Any]) -> dict[str, Any]:
    after = model.cost()
    result = dict(after)
    for key in ("calls", "input_tokens", "output_tokens", "errors"):
        result[key] = after[key] - before[key]
    rates = after.get("rates_per_million")
    result["usd"] = None if not rates else (
        result["input_tokens"] * rates[0] + result["output_tokens"] * rates[1]
    ) / 1_000_000
    return result


def make_system(name: str, conv_id: str):
    if name == "meshmind":
        from systems.meshmind_system import MeshMindSystem
        return MeshMindSystem(RUN_ROOT, conv_id)
    if name == "vector_rag":
        from systems.vector_rag_system import VectorRagSystem
        return VectorRagSystem(RUN_ROOT, conv_id)
    if name == "mem0":
        from systems.mem0_system import Mem0System
        return Mem0System(RUN_ROOT, conv_id)
    if name == "zep":
        from systems.zep_system import ZepSystem
        return ZepSystem(RUN_ROOT, conv_id)
    raise ValueError(name)


def estimate_run(systems: list[str], conversations: list[Conversation], answerer: str, judge: str) -> float:
    n_qa = sum(len(conv.qa) for conv in conversations) * len(systems)
    # Context is system dependent; 3k input + 40 output is deliberately conservative.
    total = 0.0
    if answerer in PRICES:
        inp, out = PRICES[answerer]
        total += n_qa * (3_000 * inp + 40 * out) / 1_000_000
    if judge in PRICES:
        inp, out = PRICES[judge]
        total += n_qa * (250 * inp + 50 * out) / 1_000_000
    if "mem0" in systems:
        turns = sum(sum(len(s.turns) for s in conv.sessions) for conv in conversations)
        total += turns * (700 * 0.15 + 60 * 0.60) / 1_000_000
    return total


def run_predictions(system_name: str, conv: Conversation, model: ModelClient) -> Path:
    out = RUN_ROOT / system_name / f"{conv.sample_id}.jsonl"
    existing = read_rows(out)
    completed = {int(row["idx"]) for row in existing}
    if len(completed) == len(conv.qa):
        print(f"  [{system_name}] predictions complete; resume skip")
        return out
    before = model.cost()
    system = make_system(system_name, conv.sample_id)
    try:
        system.ingest(conv)
        started = time.time()
        for idx, qa in enumerate(conv.qa):
            if idx in completed:
                continue
            retrieval = system.retrieve(qa.question)
            prompt = ANSWER_PROMPT.format(context=retrieval.context, question=qa.question)
            try:
                prediction = model.generate(prompt)
                error = None
            except Exception as exc:
                prediction, error = "", f"{type(exc).__name__}: {exc}"
            append_row(out, {
                "schema_version": 1, "system": system_name, "conv": conv.sample_id,
                "idx": idx, "question": qa.question, "gold": qa.answer,
                "category": qa.category, "evidence": qa.evidence,
                "ctx": retrieval.context, "ctx_stats": retrieval.metadata,
                "pred": prediction, "answer_error": error,
                "answerer": model.spec, "prompt_sha256": __import__("hashlib").sha256(
                    ANSWER_PROMPT.encode()).hexdigest(),
            })
            if (idx + 1) % 20 == 0:
                print(f"  [{system_name}] answered {idx + 1}/{len(conv.qa)}", flush=True)
        system_cost = system.cost_record()
        (out.parent / f"{conv.sample_id}.system_cost.json").write_text(
            json.dumps(system_cost, indent=2) + "\n")
        (out.parent / f"{conv.sample_id}.answer_cost.json").write_text(
            json.dumps(usage_delta(model, before), indent=2) + "\n")
        print(f"  [{system_name}] predictions finished in {time.time() - started:.1f}s")
    finally:
        system.close()
    return out


def run_judging(system_name: str, conv: Conversation, judge: ModelClient) -> Path:
    source = RUN_ROOT / system_name / f"{conv.sample_id}.jsonl"
    out = RUN_ROOT / system_name / f"{conv.sample_id}.judged.jsonl"
    predictions = read_rows(source)
    existing = read_rows(out)
    completed = {int(row["idx"]) for row in existing}
    if len(completed) == len(predictions):
        print(f"  [{system_name}] judgments complete; resume skip")
        return out
    before = judge.cost()
    for row in predictions:
        idx = int(row["idx"])
        if idx in completed:
            continue
        gold = gold_string(row["gold"])
        pred = row.get("pred") or ""
        prompt = JUDGE_PROMPT.format(question=row["question"], gold=gold, prediction=pred)
        try:
            result = json.loads(judge.generate(prompt, json_schema=JUDGE_SCHEMA))
            label = result.get("label")
            if label not in {"correct", "partial", "wrong"}:
                raise ValueError(f"invalid label {label!r}")
            reason, error = str(result.get("reason", "")), None
        except Exception as exc:
            label, reason, error = "wrong", "judge call failed", f"{type(exc).__name__}: {exc}"
        append_row(out, {
            **row, "gold_str": gold, "f1": token_f1(pred, gold),
            "bleu1": bleu1(pred, gold), "judge_label": label,
            "judge_reason": reason, "judge_error": error, "judge": judge.spec,
        })
        if (idx + 1) % 20 == 0:
            print(f"  [{system_name}] judged {idx + 1}/{len(predictions)}", flush=True)
    (out.parent / f"{conv.sample_id}.judge_cost.json").write_text(
        json.dumps(usage_delta(judge, before), indent=2) + "\n")
    return out


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SUPPORTED_SYSTEMS))
    parser.add_argument("--convs", default="all")
    parser.add_argument("--answerer", choices=ANSWERERS, required=True)
    parser.add_argument("--judge", choices=JUDGES, required=True)
    parser.add_argument("--cost-cap", type=float, default=30.0)
    args = parser.parse_args()
    systems = parse_csv(args.systems)
    unknown = [name for name in systems if name not in SUPPORTED_SYSTEMS]
    if unknown:
        parser.error(f"unknown systems: {', '.join(unknown)}")
    try:
        conversations = select_conversations(args.convs, load())
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    estimate = estimate_run(systems, conversations, args.answerer, args.judge)
    print(f"Estimated upper-bound provider cost: ${estimate:.2f} (cap ${args.cost_cap:.2f})")
    if estimate > args.cost_cap:
        print("Refusing to start: estimate exceeds --cost-cap", file=sys.stderr)
        return 2

    answerer, judge = ModelClient(args.answerer), ModelClient(args.judge)
    failures: list[str] = []
    for conv in conversations:
        print(f"[{conv.sample_id}] {len(conv.qa)} QAs")
        for system in systems:
            try:
                run_predictions(system, conv, answerer)
                run_judging(system, conv, judge)
            except Exception as exc:
                message = f"{system}/{conv.sample_id}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"ERROR {message}", file=sys.stderr)

    run_cost = {"answerer": answerer.cost(), "judge": judge.cost(), "failures": failures}
    run_cost["usd"] = sum(x.get("usd") or 0 for x in (run_cost["answerer"], run_cost["judge"]))
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "last_run_cost.json").write_text(json.dumps(run_cost, indent=2) + "\n")
    print(f"Actual metered answer+judge cost: ${run_cost['usd']:.4f}")
    if failures:
        print(f"Completed with {len(failures)} system failure(s); see last_run_cost.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
