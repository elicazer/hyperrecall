# MeshMind × LoCoMo Benchmark

Evaluating MeshMind on the [LoCoMo](https://github.com/snap-research/locomo)
benchmark (ACL 2024): "Evaluating Very Long-Term Conversational Memory of LLM
Agents" — Maharana et al.

## Why LoCoMo

- Public, peer-reviewed, non-trivial (10 convos × ~590 turns × ~200 QAs each).
- Both Mem0 and Zep publish LoCoMo scores. Direct apples-to-apples comparison.
- 5 question categories test different memory skills:
  - 1 = single-hop retrieval
  - 2 = multi-hop reasoning across sessions
  - 3 = temporal reasoning (dates, order)
  - 4 = open-domain / commonsense
  - 5 = adversarial (question is unanswerable from context)

## Dataset stats

10 conversations, 5,882 total turns, 1,986 QA pairs:

| Category | Count |
|----------|-------|
| 1 (single-hop) | 282 |
| 2 (multi-hop)  | 321 |
| 3 (temporal)   | 96  |
| 4 (open-domain)| 841 |
| 5 (adversarial)| 446 |

Dataset lives at `repo/data/locomo10.json` (cloned from snap-research/locomo).

## Plan

Two evaluated systems, same input, same questions, same judge:

1. **MeshMind** — real ingest via Bedrock extractor (PR #1), retrieval via
   spreading activation over the hyperedge mesh, answer via Opus with retrieved
   subgraph in context.
2. **Baseline** — naive vector RAG over raw turns (top-k cosine on
   turn embeddings). Same LLM, same prompt, only the retrieval layer differs.

Optionally later: **full-context** (whole conversation dumped in) and
**session-summary RAG** (from the paper's `session_summary` field) as
additional reference points — the paper reports numbers for these.

## Metrics

Following the LoCoMo paper:

- **F1 (token overlap)** and **BLEU-1** on generated answers vs gold.
- **LLM-judge accuracy**: prompt Opus to score generated vs gold as
  correct / partial / wrong (Mem0/Zep both do this).
- Report per-category breakdown — spreading activation should shine on
  cat 2 (multi-hop) and cat 3 (temporal) if the hypothesis holds.

## Files (planned)

- `harness/load.py`         load locomo10.json → normalized episodes
- `harness/systems.py`      MeshMind / vector-baseline wrappers
- `harness/judge.py`        LLM-judge scorer (Bedrock Opus)
- `harness/run.py`          main entry: `python run.py --system meshmind --limit 1`
- `results/`                jsonl per-run outputs, per-category summary

## How to run (tomorrow)

```bash
cd ~/projects/meshmind/benchmarks/locomo
python -m harness.run --system meshmind --limit 1       # smoke test on 1 convo
python -m harness.run --system meshmind                 # full 10 convos
python -m harness.run --system vector-baseline
python -m harness.judge --a runs/meshmind.jsonl --b runs/vector.jsonl
```

## Estimated cost (Bedrock Opus)

Rough per-conversation:
- Ingest: ~590 turns × ~200 in-tokens each ≈ 120k in / ~15k out → ~$2.
- Retrieval + answer for ~200 QAs: ~500 in-tokens / ~50 out each → ~$1.
- Full 10 convos ≈ **$30–$40**. Judge pass adds ~$5.

Sanity: run `--limit 1` first, confirm numbers land in the same ballpark,
then batch overnight.
