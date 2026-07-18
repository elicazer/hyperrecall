# Quickstart

## Install

HyperRecall isn't on PyPI yet. Install from source:

```bash
git clone https://github.com/eliazer/hyperrecall
cd hyperrecall
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Core runtime dependencies are just `numpy`, `pyyaml`, and `typer` — storage is
the Python stdlib `sqlite3`.

## Your first mesh

```python
from hyperrecall import Mesh

mesh = Mesh(":memory:")          # in-memory; use Mesh("./mesh.db") to persist

# Ingest some conversation turns.
mesh.remember(
    "Eli mentioned he's applying to TEDx San Joaquin Hills on Aug 22, 2026",
    participants=["Eli", "David"],
    context={"topic": "TEDx", "session": "abc123"},
)
mesh.remember("HyperRecall is a hypergraph memory system", context={"topic": "HyperRecall"})

# Recall a connected subgraph relevant to a query.
result = mesh.recall("TEDx applications", budget_tokens=500)

print(result.to_context_string())   # compact, prompt-ready
print(result.to_markdown())         # human-readable, with relations + flags
```

`result` is a `Subgraph`, not a list of chunks. It exposes:

- `result.nodes` — ranked `ScoredNode`s (`.node`, `.score`, `.hop`, `.superseded`, `.contradicted_by`)
- `result.hyperedges` — the relations connecting them
- `result.to_markdown()` / `result.to_context_string()`
- `result.has_conflicts()`, `result.node_ids()`

## Contradictions and supersession

```python
a = mesh.remember("The event is in Newport", context={"topic": "TEDx"})
b = mesh.remember("The event is in Irvine", context={"topic": "TEDx"})
mesh.contradict(a.id, b.id, note="venue conflict")

for na, nb, edge in mesh.contradictions():
    print(na.text, "<>", nb.text)

old = mesh.remember("Event on Aug 20", context={"topic": "TEDx"})
new = mesh.remember("Event on Aug 22", context={"topic": "TEDx"})
mesh.supersede(old.id, new.id)       # newest preferred; old kept but flagged
```

## Introspection

```python
info = mesh.inspect_node(a.id)
# {id, text, kind, confidence, activation, access_count, created_at, metadata, edges:[...]}
print(mesh.stats())                  # {'nodes': N, 'hyperedges': M, 'members': K}
```

## Portable export / import

```python
mesh.export("./my_memory")           # directory of Markdown + YAML
restored = Mesh.import_dir("./my_memory", ":memory:")
```

## CLI

```bash
hyperrecall remember "Eli is building HyperRecall" -p Eli --topic HyperRecall --db mesh.db
hyperrecall recall "what is hyperrecall" --db mesh.db --budget 300
hyperrecall export mesh.db ./export
hyperrecall import ./export restored.db
hyperrecall stats mesh.db
hyperrecall demo
```

## Custom embeddings and decay

```python
import numpy as np
from hyperrecall import Mesh

def my_embed(text: str) -> np.ndarray:
    ...  # call your favourite embedding model, return a 1-D float vector

mesh = Mesh("./mesh.db", embed=my_embed, decay_curve="power_law")
```
