# Results

| System | Correct | Score |
|---|---:|---:|
| MeshMind `v2-moat` | 10 / 10 | 100% |
| Vector-RAG top-2 | 0 / 10 | 0% |

MeshMind passed all five current-state questions by removing the superseded or contradicted edge and annotating the survivor. It passed all five change-history questions by returning both statements with explicit `before` and `after` labels.

The vector baseline retrieved passages independently. It could not satisfy the current-state clean-context requirement when stale text was also relevant, and it could not create structural `before`/`after` labels for history questions. Detailed contexts and booleans are recorded in `results.json`.
