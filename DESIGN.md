# HyperRecall Design

This document explains what HyperRecall is, why it is built on a hypergraph rather
than a knowledge graph or a vector store, and how each subsystem works. It is
the canonical record of the design decisions made while scaffolding v0.0.1.

---

## 1. The problem with current AI memory

LLM agents have no long-term memory of their own. The dominant workarounds all
lose something important:

**Context compaction loses too much.** When a conversation outgrows the context
window, the transcript is summarized and the original is discarded. Summaries
are lossy by construction: they keep the gist and throw away the specifics —
the exact date, the offhand preference, the fact that two statements were made
in the same breath. You cannot recover what compaction deleted.

**Vector databases have no structure.** The standard RAG pattern — embed chunks,
retrieve by cosine similarity — treats memory as an unordered bag. There is no
time, no causality, no notion that memory A contradicts memory B or that C
replaced D. Retrieval returns a flat list of nearby chunks with no relationships
between them. The model has to reconstruct structure that was never stored.

**Knowledge graphs are lossy in a subtler way.** KG-based memory systems improve
on vectors by adding structure — but they model that structure as
`(head, relation, tail)` **triples**. A triple is a binary edge: it can only
ever relate *two* things. Real episodes relate *many* things at once, and
shredding an episode into binary edges destroys the very co-occurrence that made
it a memory. More on this below, because it is the crux of HyperRecall.

HyperRecall's bet: **the unit of memory is the episode, and the episode is
inherently N-ary.** The right substrate is a hypergraph.

---

## 2. Why hypergraphs specifically

A **hypergraph** generalizes a graph: an edge (a *hyperedge*) can connect any
number of nodes, not just two. In HyperRecall a hyperedge is a first-class object
with an id, a type, an activation weight, a decay rate, provenance, metadata,
and a set of **members**, where each member is a node paired with a *role* and a
*weight*.

### A concrete example

Suppose the agent observes:

> "On July 13 at 8pm, Eli asked David about TEDx applications; they discussed the
> Newport event."

The episode relates **five** things: a *person who asked* (Eli), a *person
asked* (David), a *topic* (TEDx applications), a *place/event* (the Newport
event), and a *time* (Jul 13, 8pm). In HyperRecall this is **one** `Experience`
hyperedge:

```
Experience (edge_ab12cd34)
├── role=asker      → node: "Eli"
├── role=asked      → node: "David"
├── role=topic      → node: "TEDx applications"
├── role=event      → node: "the Newport event"
└── role=time       → node: "2026-07-13T20:00"
```

Everything about the episode is captured in a single object. Retrieval that
touches *any* participant can recover the *whole* episode with its roles intact.

### How a knowledge graph would model the same thing (badly)

A triple store must break the episode into binary edges. Something like:

```
(Eli)            —asked→        (David)
(conversation_1) —about→        (TEDx applications)
(conversation_1) —mentions→     (Newport event)
(conversation_1) —hasParticipant→ (Eli)
(conversation_1) —hasParticipant→ (David)
(conversation_1) —occurredAt→   (2026-07-13T20:00)
```

Look at what happened:

1. **A synthetic "conversation_1" node had to be invented** to hang the
   participants off of — the KG needs a hub node to fake N-ary structure. This
   *reification* is exactly the workaround that proves triples can't do the job
   natively.
2. **The roles are smeared across relation names** (`asked`, `about`,
   `hasParticipant`). To reconstruct "who asked whom about what," a consumer must
   join six rows and hope the relation vocabulary is consistent.
3. **Co-occurrence is gone.** Nothing records that these six edges are *one*
   event. Retrieve "Newport event" and you get the `mentions` edge; the fact
   that Eli and David were the people talking about it is two more joins away and
   might be pruned by a top-k cutoff.

Reification is essentially a hand-rolled, lossy hyperedge. HyperRecall makes the
hyperedge the primitive, so none of this is necessary. Arity is arbitrary, roles
are structural (not baked into relation strings), and the episode stays whole.

This is the whole point of the project. If we ever collapsed hyperedges to
`(head, relation, tail)` we would have built a knowledge graph and thrown away
the reason to exist — which is why `test_hypergraph.py` asserts that a real
hyperedge of **arity ≥ 3** exists in the running system.

---

## 3. Spreading-activation retrieval

Retrieval is modeled on spreading activation (Collins & Loftus, 1975), a
cognitive theory of how humans recall: a cue activates a few concepts, and that
activation spreads along associations, weakening with distance.

### Algorithm walkthrough

```
recall(query, k_hops, budget):
    # 1. SEED — find where to start.
    q_vec  = embed(query)
    seeds  = semantic_search(q_vec)      ∪  fts_search(query)   # {node_id: energy}

    # 2. SPREAD — propagate energy through hyperedges.
    scores  = defaultdict(0)
    frontier = seeds (biased by each node's live activation)
    for hop in 1..k_hops:
        next_frontier = defaultdict(0)
        for (src, energy) in frontier:
            if energy < min_energy: continue
            for edge in hyperedges_touching(src):
                conducted = energy * edge.activation_weight * hop_decay
                for member in edge.members where member != src:
                    delivered = conducted * member.weight * src.weight
                    scores[member]        += delivered
                    next_frontier[member] += delivered
        frontier = next_frontier

    # 3. ASSEMBLE — build a connected subgraph.
    nodes = [node(id) for id in scores]
    annotate_supersession(nodes)   # flag nodes a Supersedes edge points away from
    annotate_contradiction(nodes)  # flag nodes joined by a Contradicts edge
    rank nodes by (score, confidence); sink superseded if prefer_newest
    trim to budget_tokens
    reinforce(kept nodes)          # Hebbian: accessing a memory strengthens it
    return Subgraph(nodes, edges)
```

The defining property is step 2's inner loop: **energy entering one member of a
hyperedge is delivered to *all* the other members simultaneously.** Seed "the
Newport event" and, in a single hop, Eli, David, the topic and the timestamp all
light up because they share one `Experience` edge. A binary-edge graph would need
multiple hops and a hub node to achieve the same, attenuating the signal.

Energy attenuates via three multiplicative factors — the edge's
`activation_weight`, the member `weight` (role coupling), and a per-hop
`hop_decay` — and the frontier is pruned below `min_energy`, keeping the spread
local and cheap. The output is a **connected subgraph**, not a ranked list of
chunks, so the model receives memory *with its structure attached*.

---

## 4. Storage schema

HyperRecall persists to a **single SQLite file** (`hyperrecall/storage/schema.sql`).
The tables:

- **`nodes`** — id, text, kind, confidence, decay_rate, created_at, metadata(JSON).
  Node content is treated as immutable.
- **`hyperedges`** — id, type, activation_weight, decay_rate, confidence,
  created_at, provenance(JSON), metadata(JSON).
- **`hyperedge_nodes`** — the join table that makes hyperedges *real*:
  `(hyperedge_id, node_id, role, weight)`, one row per membership. N rows per
  edge ⇒ arbitrary arity, per-node roles.
- **`activations`** — `(node_id, base, updated_at, access_count)`. Kept separate
  from `nodes` so decay/reinforcement never rewrite immutable content. Live
  activation is computed on read as `decay(base, now - updated_at, rate)`.
- **`embeddings`** — `(node_id, dim, vector)` where `vector` is a little-endian
  **float32 BLOB**.
- **`nodes_fts`** — an FTS5 virtual table over node text, kept in sync by
  triggers, for lexical seed discovery.

### Design choice: embeddings as BLOBs, not sqlite-vss

The spec offered a choice between `sqlite-vss` and storing embeddings as blobs
with numpy search. **We chose BLOBs + numpy.** Rationale:

- **Portability first.** A HyperRecall database must be a single file that opens
  anywhere with the Python stdlib `sqlite3` — no compiled extension to install,
  version-match, or ship per-platform. `sqlite-vss` is a native extension and
  would break the "one portable file, zero native deps" promise.
- **Scale is not the current bottleneck.** For the corpus sizes an agent's
  working memory holds (thousands to low-hundred-thousands of nodes), a numpy
  matrix-vector product is milliseconds. ANN indexing matters at millions of
  vectors — a later optimization, not a v0 requirement.
- **The embedder is pluggable anyway.** The default `hash_embed` is a
  deterministic, dependency-free hashing embedder (reproducible tests, honest
  about not being semantic). Swap in a real model via `Mesh(embed=...)`.

---

## 5. Decay and reinforcement math

Human memory forgets what it doesn't use and strengthens what it does. HyperRecall
models both with small, pluggable functions in `hyperrecall/decay.py`.

**Decay (forgetting).** The default is an Ebbinghaus-style exponential curve:

```
retention(t) = exp(-decay_rate · elapsed_days)
live_activation = base · retention(now - updated_at)
```

`decay_rate` is per-day; `0` means never forget. Alternatives ship in a registry
(`power_law` — a better long-horizon fit; `linear` — for debugging) and any
`DecayFn` can be supplied. `half_life_to_rate(days)` converts an intuitive
half-life into a rate.

**Reinforcement (Hebbian).** Accessing a memory boosts it, with diminishing
returns toward a ceiling so hot nodes don't run away:

```
reinforce(a) = a + amount · (ceiling - a) / ceiling
```

Recall reinforces every node it returns, so frequently-recalled memories resist
forgetting — precisely the desirable feedback loop. The `activations` table
records the post-boost `base` and a fresh `updated_at`; subsequent reads decay
from there.

---

## 6. Contradiction and supersession semantics

These are dedicated hyperedge types with retrieval-time meaning.

**Contradiction.** A `Contradicts` hyperedge (members in role `claim`) links two
or more mutually-exclusive nodes. Retrieval **surfaces all of them** and flags
each with `contradicted_by` (and a `[CONFLICT]` marker in the context string).
HyperRecall deliberately does *not* auto-resolve the conflict — the agent is better
placed to decide, and silently picking one is how KG memory loses information.
`mesh.contradictions()` enumerates every conflicting pair with its edge.

**Supersession.** A `Supersedes` hyperedge (roles `old` and `new`) records that
newer information replaces older. By default `prefer_newest=True` sinks the
superseded node in ranking and flags it `superseded` / `[OUTDATED]` — **but it is
still retrievable**, so history is never destroyed. Set `prefer_newest=False` to
weigh purely by activation. This gives agents both "what's true now" and "what we
used to believe," which matters for auditing and for reasoning about change.

---

## 7. Portable file format

Any mesh exports to a directory:

```
export/
  manifest.yaml        # schema_version, format, counts
  nodes/<id>.md        # YAML frontmatter (structure) + body (node text)
  edges/<id>.md        # YAML frontmatter carrying members[], roles, weights
```

**One file per object** (documented choice): it makes git diffs meaningful, lets
a human hand-edit a single memory, and lets external tools contribute a memory by
dropping in a file. Node *content* lives in the Markdown body; all structure,
activations, decay parameters, provenance and metadata live in frontmatter.

**Losslessness.** Round-trip (`export → import`) reproduces the graph exactly:
ids, text, kinds, confidences, decay rates, timestamps, metadata, edge types,
member roles and weights, and the full activation state (base, updated_at,
access_count) all survive. Embeddings are the one thing **not** written to disk —
they are recomputed from node text on import. With the default deterministic
embedder this is bit-for-bit lossless; with a custom embedder, import re-embeds
from text. The graph structure that defines the memory is always exact.
`test_portable.py` verifies the round-trip, including a high-arity edge.

This is the "USB-C for AI memory" thesis: memory that isn't trapped in one
vendor's store.

---

## 8. What we deliberately don't do (yet)

Scope discipline for v0.0.1. Explicitly out of scope, by decision:

- **LLM-based ingestion.** Extraction is a documented heuristic stub
  (`ingest/extractor.py`): one statement node, one node per participant, salient
  context keys promoted to nodes, all bound by one `Experience` edge. It already
  produces genuine arity-≥3 hyperedges. LLM decomposition into atomic,
  well-typed nodes/edges is the top roadmap item.
- **Real embedding models.** Default is `hash_embed` (deterministic, non-semantic,
  great for reproducible tests). Real models are pluggable but not bundled — no
  heavy ML deps in core.
- **Entity resolution / dedup.** Two turns mentioning "Eli" currently create two
  entity nodes. Canonicalization comes with LLM ingestion.
- **Automatic contradiction/supersession detection.** The *edges* and their
  retrieval semantics are implemented; *deciding* when to add them is left to the
  caller (or a future LLM ingest step).
- **Federated sync, encryption, cloud.** HyperRecall is a local, single-file library.
  The portable format is the intended sync substrate; encryption and multi-agent
  sync are future work.

Everything in scope works end-to-end and is covered by tests.
