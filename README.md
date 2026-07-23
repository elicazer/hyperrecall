# HyperRecall

**HyperRecall gives AI agents memory that works the way brains do — not a bag of chunks, a web of experiences.**

Most "AI memory" today is a vector database with a nice API: you embed text, you
store chunks, you retrieve the nearest ones. It has no structure, no notion of
time, no idea that two memories contradict each other, and no concept of one
fact making another obsolete. When your context window fills up and gets
compacted, the nuance is gone for good.

HyperRecall is different. It stores memory as a **hypergraph** — first-class
relations that bind *many* things at once (a person, a project, a decision, an
outcome, a moment in time) — and retrieves it with **spreading activation**, the
same mechanism cognitive scientists use to model human recall. Memories
**decay** when unused, **strengthen** when accessed, can **contradict** each
other, and can be **superseded** by newer information. And the whole thing
exports to a directory of plain Markdown files, so your agent's memory can move
between tools.

> ⚠️ **Pre-release (v0.0.1).** This is the reference implementation and the
> substrate is real and tested end-to-end, but the ingestion pipeline is a
> heuristic stub (LLM-based extraction is on the roadmap). Not on PyPI yet.

---

## Quickstart

```bash
git clone https://github.com/elicazer/hyperrecall
cd hyperrecall
pip install -e .
# (aspirational, once published: pip install hyperrecall)
```

```python
from hyperrecall import Mesh

mesh = Mesh(":memory:")                       # or Mesh("./mesh.db")
mesh.remember("Eli is building HyperRecall", participants=["Eli"], context={"topic": "HyperRecall"})
mesh.remember("HyperRecall uses hypergraphs", context={"topic": "HyperRecall"})

result = mesh.recall("what is hyperrecall", budget_tokens=300)
print(result.to_context_string())             # prompt-ready memory
```

Try the demos:

```bash
python examples/hello_world.py         # ingest → recall → inspect, with decay
python examples/contradiction_demo.py  # conflict detection
python examples/portable_export.py     # export to Markdown and re-import
hyperrecall demo                          # the CLI does the same
```

---

## Why not just use Mem0 / Zep?

An honest comparison. These are good tools; HyperRecall makes different bets.

| | Vector memory (naive RAG) | Mem0 / Zep (KG memory) | **HyperRecall** |
|---|---|---|---|
| Structure | none — flat chunks | knowledge graph: `(head, relation, tail)` triples | **hypergraph: N-ary edges with roles** |
| One "Eli asked David about TEDx on Jul 13" fact | 1 opaque chunk | ~4–6 lossy triples that lose the *co-occurrence* | **1 `Experience` edge binding all 5 participants** |
| Retrieval | top-k cosine | graph walk / triple lookup | **spreading activation → connected subgraph** |
| Forgetting | none (or crude TTL) | usually none | **pluggable decay curve (Ebbinghaus, power-law)** |
| Reinforcement on access | none | none | **Hebbian boost** |
| Contradictions | invisible | often silently overwritten | **explicit `Contradicts` edge, both surfaced with a flag** |
| Supersession | invisible | overwrite (history lost) | **`Supersedes` edge; newest preferred, history kept** |
| Portability | proprietary store | proprietary store | **directory of Markdown+YAML, lossless round-trip** |
| Core deps | vector DB | vector DB + graph DB + LLM | **stdlib sqlite3 + numpy** |

The core disagreement is **triples vs. hyperedges**. A knowledge graph shreds
"Eli asked David about TEDx applications on July 13 at 8pm" into a handful of
binary edges (`Eli —asked→ David`, `conversation —about→ TEDx`, …). The fact
that these all happened *in one episode* — the thing a human actually
remembers — is exactly what gets lost. HyperRecall keeps the episode whole as a
single hyperedge. See [`DESIGN.md`](DESIGN.md) for the full argument.

---

## Design principles

1. **Genuine hypergraph.** Hyperedges are first-class objects with a type,
   weight, decay rate, provenance, and members that each carry a *role*. Arity
   is arbitrary (N ≥ 2). This is not a triple store wearing a costume.
2. **Neuroscience-inspired.** Spreading activation, forgetting curves, Hebbian
   reinforcement, contradiction and supersession — memory as a dynamic system,
   not a static index.
3. **Portable.** Any mesh exports to human-readable Markdown+YAML and imports
   back losslessly. Your memory is yours; move it between Claude, Cursor,
   OpenClaw, or your own scripts. "USB-C for AI memory."
4. **Open-source forever.** Apache 2.0. Python-first, TS SDK later. No cloud
   lock-in, no proprietary format.

---

## How it works (30 seconds)

- **Nodes** are memory units (facts, entities, decisions, outcomes), each with a
  `confidence`, an `activation` (its live salience), and a `decay_rate`.
- **Hyperedges** connect N ≥ 2 nodes; each member has a `role` and a `weight`.
  Types include `Experience`, `Contradicts`, `Supersedes`, `Refines`,
  `CausedBy`, `MentionedTogether`.
- **Recall** embeds your query, finds seed nodes (semantic + lexical), then
  spreads activation through hyperedges for *k* hops. Energy entering one member
  of a hyperedge lights up *all* the others. You get back a **connected
  subgraph**, ranked, with conflicts flagged — rendered to Markdown or a compact
  context string.
- **Storage** is a single SQLite file (FTS5 for lexical search, float32 BLOB
  embeddings searched in numpy — no native extensions required).

---

## Roadmap

- [x] Hypergraph substrate (nodes, N-ary hyperedges, roles)
- [x] Spreading-activation retrieval → subgraph
- [x] Decay + Hebbian reinforcement
- [x] Contradiction + supersession semantics
- [x] Portable Markdown+YAML export/import (lossless)
- [x] SQLite + FTS5 storage, numpy embedding search
- [x] CLI (`hyperrecall remember | recall | export | import | demo`)
- [ ] **LLM-based ingestion** — extract atomic nodes + typed edges from raw turns
- [ ] Pluggable real embedding models (OpenAI, local sentence-transformers)
- [ ] Automatic contradiction/supersession detection at ingest
- [ ] TypeScript SDK reading the same portable format
- [ ] Entity resolution / node dedup
- [ ] Optional encryption + federated sync

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [`LICENSE`](LICENSE). Built by Eli Azer.
