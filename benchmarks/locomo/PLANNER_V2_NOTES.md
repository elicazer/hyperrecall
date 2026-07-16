# Query Planner v2

Planner v2 is an opt-in query-time reasoning layer. The existing
`mesh.recall(question)` path remains unchanged; applications enable the new
path with `mesh.recall(question, plan="v2")`.

## Design

The planner performs four stages:

1. **Classify.** Gemini 2.5 Pro returns one of `single_hop`, `multi_hop`,
   `temporal`, `open_domain`, or `adversarial`, together with entities, time
   constraints, and question kind. Calls use `google-generativeai` and retry
   with exponential backoff. An offline heuristic is available for tests and
   environments without a Gemini key.
2. **Retrieve.** Single-hop queries anchor on entity nodes and their incident
   edges. Multi-hop queries are decomposed by Gemini into two or three atomic
   questions, retrieved independently, and joined on shared participants.
   Temporal, open-domain, and adversarial queries begin with the established
   semantic-seed/spreading-activation retrieval.
3. **Resolve.** `Supersedes`/`Supersession` edges remove older claims.
   `Contradicts`/`Contradiction` groups retain the newest timestamped claim.
   Questions explicitly asking what was previously believed retain both sides
   and mark them `HISTORICAL_CONFLICT`. Temporal constraints are then applied
   to edge timestamps.
4. **Assemble.** `PlannedRecall.results` contains each selected hyperedge, its
   materialized participants and roles, normalized timestamp, provenance, and
   annotations. `to_context_string()` renders these records for an answerer.

No new edge types are introduced. The implementation accepts both the
repository's canonical plural relation names and singular aliases found in
older data.

## Running LoCoMo

```bash
python benchmarks/locomo/run_mesh_phase2.py --planner v2
```

This writes predictions to `runs/planner_v2/conv-26.meshmind.jsonl`. The run
uses Gemini 2.5 Pro for planning and the existing Gemini answerer. Evaluation
results and comparison with `runs/phase3_baseline/summary.json` are recorded in
`runs/planner_v2/summary.json`.

## Verification

```bash
python -m meshmind.query.planner --demo
pytest -q
```

The demo builds an in-memory mock graph and exercises all five routing classes.
