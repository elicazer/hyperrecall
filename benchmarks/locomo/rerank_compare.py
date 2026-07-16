"""Quota-light comparison of v2-rerank vs the v2 baseline (integration_v2).

Computes token-F1 / BLEU-1 locally (no LLM) per LoCoMo category on the set of
question indices present in BOTH run files, so a partial rerank run still yields
an apples-to-apples delta. If both runs carry LLM judge labels
(``judge_label``), those accuracies are reported too.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_judge import token_f1, bleu1  # noqa: E402

CATS = {"1": "single-hop", "2": "multi-hop", "3": "temporal",
        "4": "open-domain", "5": "adversarial"}


def load_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.open():
        r = json.loads(line)
        if "429" in r.get("pred", "") or "ANSWER_ERROR" in r.get("pred", ""):
            continue  # skip rows whose answer never completed
        rows[r["idx"]] = r
    return rows


def summarize(rows: dict[int, dict], idxs: list[int]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for i in idxs:
        r = rows[i]
        by_cat[str(r["category"])].append(r)
    out = {}
    all_rows = [rows[i] for i in idxs]
    for cat in sorted(by_cat) + ["all"]:
        items = all_rows if cat == "all" else by_cat[cat]
        if not items:
            continue
        f1 = sum(token_f1(x["pred"], str(x["gold"])) for x in items) / len(items)
        bl = sum(bleu1(x["pred"], str(x["gold"])) for x in items) / len(items)
        entry = {"n": len(items), "avg_f1": round(f1, 3), "avg_bleu1": round(bl, 3)}
        labels = [x.get("judge_label") for x in items if x.get("judge_label")]
        if labels:
            entry["acc_correct"] = round(sum(l == "correct" for l in labels) / len(labels), 3)
            entry["acc_c_or_p"] = round(
                sum(l in ("correct", "partial") for l in labels) / len(labels), 3)
        out[cat] = entry
    return out


def main() -> int:
    base = load_rows(Path(sys.argv[1]))     # integration_v2 judged jsonl
    rerank = load_rows(Path(sys.argv[2]))   # rerank_v1 jsonl (or judged)
    shared = sorted(set(base) & set(rerank))
    print(f"shared completed questions: {len(shared)} "
          f"(base={len(base)} rerank={len(rerank)})\n")
    b = summarize(base, shared)
    r = summarize(rerank, shared)
    hdr = f"{'cat':12} {'n':>3}  {'F1_base':>7} {'F1_rerank':>9} {'dF1':>7}"
    print(hdr)
    for cat in sorted(set(b) - {"all"}) + ["all"]:
        if cat not in b:
            continue
        name = "OVERALL" if cat == "all" else CATS.get(cat, cat)
        df1 = r[cat]["avg_f1"] - b[cat]["avg_f1"]
        print(f"{name:12} {b[cat]['n']:>3}  {b[cat]['avg_f1']:>7.3f} "
              f"{r[cat]['avg_f1']:>9.3f} {df1:>+7.3f}")
    if any("acc_correct" in v for v in r.values()):
        print("\n(judge labels present)")
        for cat in sorted(set(b) - {"all"}) + ["all"]:
            if cat not in b or "acc_correct" not in b.get(cat, {}):
                continue
            name = "OVERALL" if cat == "all" else CATS.get(cat, cat)
            print(f"{name:12} acc {b[cat]['acc_correct']:.3f} -> "
                  f"{r[cat]['acc_correct']:.3f}")
    Path(sys.argv[3] if len(sys.argv) > 3 else "rerank_compare.json").write_text(
        json.dumps({"shared_n": len(shared), "baseline": b, "rerank": r}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
