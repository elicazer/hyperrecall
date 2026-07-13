"""Import a portable Markdown + YAML directory back into a MeshMind database.

The inverse of :mod:`meshmind.portable.exporter`. Reads every ``nodes/*.md`` and
``edges/*.md`` file, reconstructs :class:`Node` / :class:`Hyperedge` objects with
their original ids, activations, decay parameters and metadata, and writes them
into an (empty) store. Embeddings are recomputed from node text via ``embed``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..models import Hyperedge, HyperedgeMember, Node
from ..storage.embeddings import DEFAULT_DIM, EmbedFn, hash_embed
from ..storage.sqlite_store import SqliteStore


def import_dir(
    store: SqliteStore,
    directory: str | Path,
    *,
    embed: EmbedFn | None = None,
) -> dict[str, int]:
    root = Path(directory)
    if not (root / "manifest.yaml").exists():
        raise FileNotFoundError(f"{root} is not a MeshMind portable directory (no manifest.yaml)")
    embed = embed or (lambda t: hash_embed(t, DEFAULT_DIM))

    n_nodes = 0
    for md in sorted((root / "nodes").glob("*.md")):
        fm, body = _parse(md)
        node = Node(
            id=fm["id"],
            text=body,
            kind=fm.get("kind", "fact"),
            confidence=fm.get("confidence", 1.0),
            activation=fm.get("activation_base", 1.0),
            decay_rate=fm.get("decay_rate", 0.05),
            created_at=fm.get("created_at"),
            metadata=fm.get("metadata") or {},
        )
        store.add_node(node, vector=embed(node.text))
        # Restore exact activation state (base + timestamp + access count).
        store.set_activation(
            node.id,
            base=fm.get("activation_base", 1.0),
            updated_at=fm.get("activation_updated_at", node.created_at),
        )
        _restore_access_count(store, node.id, int(fm.get("access_count", 0)))
        n_nodes += 1

    n_edges = 0
    for md in sorted((root / "edges").glob("*.md")):
        fm, _ = _parse(md)
        edge = Hyperedge(
            id=fm["id"],
            type=fm["type"],
            activation_weight=fm.get("activation_weight", 1.0),
            decay_rate=fm.get("decay_rate", 0.03),
            confidence=fm.get("confidence", 1.0),
            created_at=fm.get("created_at"),
            provenance=fm.get("provenance") or {},
            metadata=fm.get("metadata") or {},
            members=[
                HyperedgeMember(m["node_id"], m.get("role", "member"), m.get("weight", 1.0))
                for m in fm.get("members", [])
            ],
        )
        store.add_hyperedge(edge)
        n_edges += 1

    return {"nodes": n_nodes, "hyperedges": n_edges}


def _restore_access_count(store: SqliteStore, node_id: str, count: int) -> None:
    store.raw().execute(
        "UPDATE activations SET access_count = ? WHERE node_id = ?", (count, node_id)
    )
    store.raw().commit()


def _parse(path: Path) -> tuple[dict, str]:
    """Split a frontmatter markdown file into (frontmatter_dict, body_text)."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path} is missing YAML frontmatter")
    _, fm_text, body = raw.split("---", 2)
    fm = yaml.safe_load(fm_text) or {}
    return fm, body.strip()
