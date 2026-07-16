# Question-Conditional Reranker (`plan="v2-rerank"`)

## Problem

Some single-hop questions failed even though the correct edge was already in the
retrieved bundle. The wrong-but-similar edge ranked higher on lexical/vector
similarity, and the answerer picked from the top.

> "What did Melanie paint recently?" retrieved both the *lake sunrise* and the
> *horse* facts, but the *horse* edge scored higher because "recently" pushed
> retrieval toward newer edges regardless of topic.

Retrieval scores proximity to the *seeds*, not relevance to the *question*. The
fix is a second pass that scores each candidate against the actual question.

## Design

Inserted in `QueryPlanner.recall` **between** candidate retrieval (spreading
activation + neighbourhood expansion + supersession/temporal filtering) and the
answerer-facing assembly (`_assemble`). Old `plan="v2"` is byte-for-byte
unchanged; the rerank only runs when `recall(..., rerank=True)`.

Pipeline (`_rerank_subgraph` → `rerank_candidates`):

1. **Candidates.** `_build_candidates` turns the retrieved hyperedges into
   `RerankCandidate`s. Each edge inherits a retrieval score = the max
   `rank_score` (or `score`) over its member nodes. Candidates are sorted by
   retrieval score and the top **`K_candidate = 25`** are kept.
2. **Text repr.** `_candidate_text` renders each edge as its `source_text`
   (or `summary`) plus `role=participant` pairs — short, question-agnostic.
3. **One batched Gemini call.** `_llm_score_batch` sends *all 25* candidates in a
   single prompt ("Question: … Rate each memory 0–10 for how directly it helps
   answer the question") and parses a JSON `{"scores": [int, …]}`. Exactly one
   call per question — never 25.
4. **Blend.** `final_score = 0.6 * llm_score/10 + 0.4 * normalized_retrieval`,
   where the retrieval score is min-maxed across the candidate set so the 0.4
   term is on the same 0–1 scale as the LLM term.
5. **Cut.** Sort by `final_score`, keep the top **`K_final = 8`**. The subgraph
   is rebuilt from just those edges (and their nodes) before assembly.

`K_candidate` / `K_final` are method args (defaults 25 / 8), threaded through
`mesh.recall(plan="v2-rerank", k_candidate=…, k_final=…)` and the benchmark
runner's `--k-candidate` / `--k-final`.

### Model

The rerank scorer is **Gemini 2.5 Flash** (per project guardrail), constructed
separately from the planner's classification client so the `v2` path is
untouched. In tests an injectable `rerank_llm` (falling back to `llm`) makes the
scoring deterministic.

### Robustness

- No `GEMINI_API_KEY` / no scorer → fall back to retrieval order (`applied:false`,
  reason `llm_unavailable`).
- Malformed JSON or wrong score count → fall back to retrieval order
  (reason `score_count_mismatch`). Never crashes.
- Non-integer scores are coerced (`_coerce_score`: `float → round → clamp 0..10`,
  garbage → 0).

### Observability

`PlannedRecall.rerank` carries the full trace: per-candidate
`retrieval_score`, `llm_score`, `final_score`, `original_rank`, `new_rank`,
`rank_delta`, `in_top_k`, plus `promoted_into_top_k` (candidates that were
outside the top-`K_final` on retrieval order but made the cut after reranking).
The benchmark surfaces these under each row's `ctx_stats.rerank_*`.

## Benchmark

Mesh: `phase1_v2` (extractor v2, 3909 nodes / 1817 edges) — the same mesh
`integration_v2` was scored on, for an apples-to-apples comparison.

```
python benchmarks/locomo/run_mesh_phase2.py --planner v2-rerank    # -> runs/rerank_v1/
python benchmarks/locomo/phase3_judge.py \
    --in-dir runs/rerank_v1 --out-dir runs/rerank_v1               # -> runs/rerank_v1/summary.json
```

Because `v2-rerank` makes two Gemini calls per question, the runner throttles
(`RERANK_SLEEP`, default 4.5s) and wraps the answerer in a 429-aware backoff so a
rate-limited run still completes.

## Baseline (integration_v2, same mesh)

| category      |  n | acc_correct | correct/partial | avg_f1 |
|---------------|----|-------------|-----------------|--------|
| 1 single-hop  | 32 | 0.281       | 0.625           | 0.232  |
| 2 multi-hop   | 37 | 0.324       | 0.649           | 0.429  |
| 3 temporal    | 13 | 0.538       | 0.692           | 0.022  |
| 4 open-domain | 70 | 0.529       | 0.700           | 0.415  |
| 5 adversarial | 47 | 0.809       | 0.809           | 0.000  |
| **overall**   |199 | **0.518**   | **0.704**       | **0.264** |

## Results (rerank_v1)

_Pending — filled in after the scored run below._

<!-- RESULTS -->
