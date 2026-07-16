"""Phase 3: judge Phase 2 outputs (auto metrics + Gemini 2.5 Pro LLM-judge)."""
from __future__ import annotations
import argparse
import json, math, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types

_ap = argparse.ArgumentParser()
_ap.add_argument("--in-dir", default=str(ROOT / "runs" / "phase2"), help="directory containing phase2 .jsonl outputs")
_ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase3"), help="directory for judged outputs + summary.json")
_ap.add_argument("--conv", default="conv-26")
_args, _ = _ap.parse_known_args()

CONV_ID = _args.conv
IN_DIR = Path(_args.in_dir)
OUT_DIR = Path(_args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_MODEL = "gemini-2.5-flash"
CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}
NL = chr(10)
_word_re = re.compile(r"\b\w+\b")

def _tokens(s):
    return _word_re.findall((s or "").lower())

def token_f1(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def bleu1(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    prec = overlap / len(p) if p else 0.0
    bp = 1.0 if len(p) >= len(g) else math.exp(1 - len(g) / max(1, len(p)))
    return bp * prec

JUDGE_PROMPT = """You are grading an answer to a question about a long-term conversation.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Grade the prediction against the gold using ONE of these labels:
  correct  - the prediction matches the gold answer in meaning (paraphrases OK).
  partial  - the prediction is on-topic and mentions the right entity/concept
             but misses part of what the gold answer says.
  wrong    - the prediction contradicts the gold or is unrelated.

If the gold is a date and the prediction is a relative expression that resolves to
the same event (e.g. "yesterday" for the correct date), mark "partial". If the gold
expects "I don't know" style refusal and the prediction refuses, mark "correct".

Return STRICT JSON: {{"label":"correct|partial|wrong","reason":"one short sentence"}}
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}

def judge_one(client, question, gold, pred):
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=JUDGE_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            if attempt == 2:
                return {"label": "wrong", "reason": "[JUDGE_ERROR: " + str(e) + "]"}
            time.sleep(1.5 * (attempt + 1))
    return {"label": "wrong", "reason": "[JUDGE_ERROR: unreachable]"}

def judge_system(client, name):
    in_path = IN_DIR / (CONV_ID + "." + name + ".jsonl")
    out_path = OUT_DIR / (CONV_ID + "." + name + ".judged.jsonl")
    rows = [json.loads(line) for line in in_path.open()]
    scores_by_cat = defaultdict(list)
    out_f = out_path.open("w")
    t0 = time.time()
    for i, r in enumerate(rows):
        raw_gold = r["gold"]
        if isinstance(raw_gold, str):
            gold = raw_gold
        elif isinstance(raw_gold, (list, tuple)):
            gold = " / ".join(map(str, raw_gold))
        else:
            gold = str(raw_gold)
        pred = r["pred"] or ""
        f1 = token_f1(pred, gold)
        b1 = bleu1(pred, gold)
        j = judge_one(client, r["question"], gold, pred)
        scored = {**r, "gold_str": gold, "f1": f1, "bleu1": b1,
                  "judge_label": j["label"], "judge_reason": j["reason"]}
        out_f.write(json.dumps(scored) + NL)
        out_f.flush()
        scores_by_cat[r["category"]].append(scored)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate else 0
            print("  [" + name + "] judged " + str(i + 1) + "/" + str(len(rows)) +
                  "  " + str(round(rate, 2)) + " q/s eta=" + str(int(eta)) + "s", flush=True)
    out_f.close()

    summary = {"system": name, "n_total": len(rows), "by_category": {}}
    total_labels = Counter()
    total_f1 = total_b1 = 0.0
    for cat, items in sorted(scores_by_cat.items()):
        labels = Counter(x["judge_label"] for x in items)
        avg_f1 = sum(x["f1"] for x in items) / len(items)
        avg_b1 = sum(x["bleu1"] for x in items) / len(items)
        strict = labels.get("correct", 0) / len(items)
        lax = (labels.get("correct", 0) + labels.get("partial", 0)) / len(items)
        summary["by_category"][cat] = {
            "name": CATEGORY_NAMES.get(cat, str(cat)),
            "n": len(items), "acc_correct": round(strict, 3),
            "acc_correct_or_partial": round(lax, 3),
            "avg_f1": round(avg_f1, 3), "avg_bleu1": round(avg_b1, 3),
            "labels": dict(labels),
        }
        total_labels.update(labels)
        total_f1 += sum(x["f1"] for x in items)
        total_b1 += sum(x["bleu1"] for x in items)
    n = len(rows)
    summary["overall"] = {
        "acc_correct": round(total_labels.get("correct", 0) / n, 3),
        "acc_correct_or_partial": round(
            (total_labels.get("correct", 0) + total_labels.get("partial", 0)) / n, 3),
        "avg_f1": round(total_f1 / n, 3),
        "avg_bleu1": round(total_b1 / n, 3),
        "labels": dict(total_labels),
    }
    return out_path, summary

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    all_summaries = {}
    for name in ("meshmind", "vector_rag"):
        in_path = IN_DIR / (CONV_ID + "." + name + ".jsonl")
        if not in_path.exists():
            print("missing " + str(in_path) + "; run phase2 first", file=sys.stderr)
            continue
        print("[judge] " + name)
        out_path, summary = judge_system(client, name)
        all_summaries[name] = summary
        print("  wrote " + str(out_path))
    (OUT_DIR / "summary.json").write_text(json.dumps(all_summaries, indent=2))

    print(NL + "=== SUMMARY ===")
    header = "category            "
    for name in all_summaries:
        header += "  " + name.rjust(36)
    print(header)
    cats = sorted({c for s in all_summaries.values() for c in s["by_category"]})
    for cat in cats:
        row = str(cat) + " " + CATEGORY_NAMES.get(cat, "").ljust(16)
        for name, s in all_summaries.items():
            b = s["by_category"].get(cat)
            if b:
                row += "  n=" + str(b["n"]).ljust(3) + " strict=" + str(b["acc_correct"]) + \
                       " lax=" + str(b["acc_correct_or_partial"]) + " F1=" + str(b["avg_f1"])
            else:
                row += "  -"
        print(row)
    print("-" * 100)
    for name, s in all_summaries.items():
        o = s["overall"]
        print("OVERALL " + name.ljust(14) + "  strict=" + str(o["acc_correct"]) +
              "  lax=" + str(o["acc_correct_or_partial"]) + "  F1=" + str(o["avg_f1"]) +
              "  BLEU1=" + str(o["avg_bleu1"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
