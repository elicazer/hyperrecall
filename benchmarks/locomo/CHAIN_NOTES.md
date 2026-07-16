# Chain planner v1

## Design

`plan="v2-chain"` reuses the v2 classifier and decomposition, then executes up
to three atomic questions in order. Each step retrieves at most eight edges and
is answered from that step's context. Every later semantic retrieval query and
step-answer prompt is prefixed with the answers learned so far. The returned
context begins with numbered `Reasoning steps`, followed by the union of the
retrieved edge contexts; the same text is available as `PlannedRecall.explanation`.

An `I don't know.` step remains in the trace, and the trace tells the final
answerer that it may override the step from the complete context. A multi-hop
classification with no sub-questions delegates to the unchanged v2 recall
path. `plan="v2"` remains routed to the original executor.

## Verification and results

- `pytest -q`: 77 passed.
- A one-question CLI smoke test reached the `v2-chain` dispatcher and wrote to
  `runs/chain_v1/`.
- The full conv-26 run and `phase3_judge` score are blocked as of 2026-07-16:
  this worktree did not contain the ignored phase-1 SQLite graph, and both
  configured Gemini credentials returned `429 RESOURCE_EXHAUSTED` while trying
  to rebuild it and in a later five-request Flash probe.

The comparison baseline in `runs/integration_v2/summary.json` has overall F1
`0.264` and multi-hop F1 `0.429`. No chain delta is reported until a valid full
run can produce `runs/chain_v1/summary.json`; recording a score from the empty
smoke database would be misleading.
