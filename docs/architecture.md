# Architecture

A layered view of the codebase. Each layer depends only on those below it.

```
                    ┌──────────────────────────────┐
   public API  ───▶ │  hyperrecall.Mesh  (mesh.py)    │
                    └──────────────┬───────────────┘
             ┌─────────────┬───────┴───────┬─────────────┐
             ▼             ▼               ▼             ▼
        ingest/       retrieval/       portable/     decay.py
      extractor.py  activation.py   exporter.py    (curves +
      (text→graph)   query.py       importer.py    reinforce)
                     (spread +      (md+yaml I/O)
                      recall)
             └─────────────┴───────┬───────┴─────────────┘
                                   ▼
                    ┌──────────────────────────────┐
   storage    ───▶  │  SqliteStore (sqlite_store)  │
                    │  schema.sql · embeddings.py  │
                    └──────────────┬───────────────┘
                                   ▼
                         SQLite file  (+ FTS5)
                                   │
                         models.py  (Node, Hyperedge, HyperedgeMember)
```

## Modules

### `models.py`
Dataclasses for the domain: `Node`, `Hyperedge`, `HyperedgeMember`, plus the
canonical hyperedge-type constants (`EXPERIENCE`, `CONTRADICTS`, `SUPERSEDES`,
`REFINES`, `CAUSED_BY`, `MENTIONED_TOGETHER`) and small JSON/id helpers. A
`Hyperedge` exposes `.arity`, `.node_ids`, and `.role_of(node_id)`.

### `storage/`
The only layer that touches SQL.
- `schema.sql` — the tables (see [DESIGN.md §4](../DESIGN.md)). Hyperedges are
  first-class; `hyperedge_nodes` gives arbitrary arity with per-member roles.
- `sqlite_store.py` — `SqliteStore`, the typed persistence API: add/get nodes and
  edges, `edges_for_node`, `edges_of_type`, `live_activation`, `reinforce_node`,
  `fts_search`, `semantic_search`.
- `embeddings.py` — the pluggable `embed` contract, the default deterministic
  `hash_embed`, BLOB (de)serialization, and numpy `cosine_search`.

### `retrieval/`
- `activation.py` — `spread()`, the k-hop spreading-activation core. Returns an
  `ActivationResult` (per-node scores, traversed edges, hop distances).
- `query.py` — `recall()` and the `Subgraph` / `ScoredNode` result types. Handles
  seed discovery, ranking, contradiction/supersession annotation, token
  budgeting, and access reinforcement.

### `ingest/`
- `extractor.py` — `extract()`, the heuristic text→hypergraph stub. Produces a
  statement node, participant/context nodes, and one binding `Experience` edge.
  The seam where LLM-based ingestion will drop in.

### `portable/`
- `exporter.py` / `importer.py` — lossless Markdown+YAML round-trip.

### `decay.py`
Pluggable forgetting curves (`exponential`, `power_law`, `linear`) and Hebbian
`reinforce()`. Selected by name via a registry or passed as a `DecayFn`.

### `mesh.py`
`Mesh` — the ergonomic public facade wiring the above together: `remember`,
`recall`, `contradict`, `supersede`, `inspect_node`, `contradictions`, `export`,
`import_dir`, `stats`.

### `cli.py`
Typer app exposing `remember`, `recall`, `export`, `import`, `stats`, `demo`.

## Data flow: a `recall()` call

1. `Mesh.recall` → `retrieval.query.recall`.
2. `_find_seeds` embeds the query and blends `semantic_search` (numpy cosine over
   embedding BLOBs) with `fts_search` (FTS5 lexical).
3. `activation.spread` propagates energy through hyperedges for *k* hops, biased
   by each node's live (decayed) activation.
4. Lit-up nodes are materialized, then annotated for supersession and
   contradiction from `Supersedes` / `Contradicts` edges.
5. Nodes are ranked, trimmed to `budget_tokens`, and reinforced (Hebbian).
6. A connected `Subgraph` (nodes + connecting edges) is returned.

See [DESIGN.md](../DESIGN.md) for the reasoning behind each choice.
