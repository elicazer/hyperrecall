# Synthetic contradiction benchmark

Vector retrieval ranks passages independently. It can retrieve both an old claim and its replacement, but it has no relation that says which one supersedes the other. This benchmark isolates that structural difference from answer-model quality.

The dataset contains 20 dated turns: ten people each make an initial statement and later change a preference, fact, plan, or status. Five questions request only the current state; five ask how the person's view changed.

The current-state rubric requires the new statement and rejects stale context containing the old one. The history rubric requires both statements plus explicit `before` and `after` annotations. MeshMind uses persisted `Supersession`/`Contradiction` hyperedges. The baseline uses transparent bag-of-words vector-style similarity and top-2 independent turns, with recency only breaking equal scores.

Run from the repository root:

```bash
/home/ubuntu/projects/meshmind/.venv/bin/python benchmarks/synthetic/contradiction/run.py
```

The script writes detailed per-question contexts to `results.json`; the checked-in aggregate is in `RESULTS.md`.
