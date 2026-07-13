"""Export a MeshMind database to a portable Markdown + YAML directory.

Layout produced::

    <dir>/
      manifest.yaml          # schema version + counts
      nodes/<node_id>.md     # one file per node (YAML frontmatter + text body)
      edges/<edge_id>.md     # one file per hyperedge (frontmatter carries members)

Why one-file-per-object (documented in DESIGN.md): it makes diffs meaningful in
git, lets a human hand-edit a single memory, and lets external tools drop new
memories in by writing a file. Node *content* lives in the markdown body;
everything structural lives in the frontmatter.

Embeddings are **not** written to disk — they are recomputed on import from the
node text via the configured embedder. With the default deterministic embedder
this is exactly lossless; with a custom embedder, re-import re-embeds. All
graph structure, activations, decay parameters and metadata round-trip exactly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..models import dumps_json, loads_json  # noqa: F401  (re-exported convenience)
from ..storage.sqlite_store import SqliteStore

SCHEMA_VERSION = 1


def export_dir(store: SqliteStore, directory: str | Path) -> Path:
    root = Path(directory)
    nodes_dir = root / "nodes"
    edges_dir = root / "edges"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    conn = store.raw()

    n_nodes = 0
    for node in store.all_nodes():
        act = conn.execute(
            "SELECT base, updated_at, access_count FROM activations WHERE node_id = ?",
            (node.id,),
        ).fetchone()
        frontmatter = {
            "id": node.id,
            "kind": node.kind,
            "confidence": node.confidence,
            "decay_rate": node.decay_rate,
            "created_at": node.created_at,
            "activation_base": (act["base"] if act else 1.0),
            "activation_updated_at": (act["updated_at"] if act else node.created_at),
            "access_count": (act["access_count"] if act else 0),
            "metadata": node.metadata,
        }
        (nodes_dir / f"{node.id}.md").write_text(
            _render(frontmatter, node.text), encoding="utf-8"
        )
        n_nodes += 1

    n_edges = 0
    for edge in store.all_hyperedges():
        frontmatter = {
            "id": edge.id,
            "type": edge.type,
            "activation_weight": edge.activation_weight,
            "decay_rate": edge.decay_rate,
            "confidence": edge.confidence,
            "created_at": edge.created_at,
            "provenance": edge.provenance,
            "metadata": edge.metadata,
            "members": [
                {"node_id": m.node_id, "role": m.role, "weight": m.weight}
                for m in edge.members
            ],
        }
        body = f"{edge.type} hyperedge connecting {edge.arity} nodes."
        (edges_dir / f"{edge.id}.md").write_text(_render(frontmatter, body), encoding="utf-8")
        n_edges += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "meshmind-portable",
        "nodes": n_nodes,
        "hyperedges": n_edges,
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return root


def _render(frontmatter: dict, body: str) -> str:
    fm = yaml.safe_dump(frontmatter, sort_keys=True, allow_unicode=True).rstrip()
    return f"---\n{fm}\n---\n\n{body}\n"
