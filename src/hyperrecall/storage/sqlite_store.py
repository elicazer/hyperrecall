"""SQLite-backed storage for the HyperRecall hypergraph.

This module owns *all* SQL. Higher layers (retrieval, portable, mesh) talk to
the graph through :class:`SqliteStore` and never touch a cursor directly.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np

from ..decay import DecayFn, current_activation, exponential_decay, reinforce
from ..models import (
    Hyperedge,
    HyperedgeMember,
    Node,
    dumps_json,
    loads_json,
    now_ts,
)
from . import embeddings as emb


def _load_schema() -> str:
    return resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")


class SqliteStore:
    """Thin, typed persistence layer over a single SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_load_schema())
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- nodes -------------------------------------------------------------
    def add_node(self, node: Node, vector: np.ndarray | None = None) -> Node:
        self._conn.execute(
            """INSERT INTO nodes(id, text, kind, confidence, decay_rate, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id,
                node.text,
                node.kind,
                node.confidence,
                node.decay_rate,
                node.created_at,
                dumps_json(node.metadata),
            ),
        )
        self._conn.execute(
            "INSERT INTO activations(node_id, base, updated_at, access_count) VALUES (?, ?, ?, 0)",
            (node.id, node.activation, node.created_at),
        )
        if vector is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, dim, vector) VALUES (?, ?, ?)",
                (node.id, int(vector.shape[0]), emb.to_blob(vector)),
            )
        self._conn.commit()
        return node

    def get_node(self, node_id: str, *, curve: DecayFn = exponential_decay) -> Node | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row, curve=curve)

    def _row_to_node(self, row: sqlite3.Row, *, curve: DecayFn) -> Node:
        act = self._conn.execute(
            "SELECT base, updated_at FROM activations WHERE node_id = ?", (row["id"],)
        ).fetchone()
        if act is None:
            live = 1.0
            last = row["created_at"]
        else:
            live = current_activation(
                act["base"], now_ts() - act["updated_at"], row["decay_rate"], curve
            )
            last = act["updated_at"]
        return Node(
            id=row["id"],
            text=row["text"],
            kind=row["kind"],
            confidence=row["confidence"],
            activation=live,
            decay_rate=row["decay_rate"],
            created_at=row["created_at"],
            last_access=last,
            metadata=loads_json(row["metadata"]),
        )

    def all_nodes(self, *, curve: DecayFn = exponential_decay) -> list[Node]:
        rows = self._conn.execute("SELECT * FROM nodes ORDER BY created_at").fetchall()
        return [self._row_to_node(r, curve=curve) for r in rows]

    # -- hyperedges --------------------------------------------------------
    def add_hyperedge(self, edge: Hyperedge) -> Hyperedge:
        if edge.arity < 2:
            raise ValueError(
                f"a hyperedge must connect >= 2 nodes (got arity {edge.arity}); "
                "HyperRecall is a hypergraph, not a set of dangling references"
            )
        self._conn.execute(
            """INSERT INTO hyperedges(id, type, activation_weight, decay_rate, confidence,
                                      created_at, provenance, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.id,
                edge.type,
                edge.activation_weight,
                edge.decay_rate,
                edge.confidence,
                edge.created_at,
                dumps_json(edge.provenance),
                dumps_json(edge.metadata),
            ),
        )
        self._conn.executemany(
            "INSERT INTO hyperedge_nodes(hyperedge_id, node_id, role, weight) VALUES (?, ?, ?, ?)",
            [(edge.id, m.node_id, m.role, m.weight) for m in edge.members],
        )
        self._conn.commit()
        return edge

    def get_hyperedge(self, edge_id: str) -> Hyperedge | None:
        row = self._conn.execute("SELECT * FROM hyperedges WHERE id = ?", (edge_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def _row_to_edge(self, row: sqlite3.Row) -> Hyperedge:
        members = self._conn.execute(
            "SELECT node_id, role, weight FROM hyperedge_nodes WHERE hyperedge_id = ?",
            (row["id"],),
        ).fetchall()
        return Hyperedge(
            id=row["id"],
            type=row["type"],
            activation_weight=row["activation_weight"],
            decay_rate=row["decay_rate"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            provenance=loads_json(row["provenance"]),
            metadata=loads_json(row["metadata"]),
            members=[HyperedgeMember(m["node_id"], m["role"], m["weight"]) for m in members],
        )

    def all_hyperedges(self) -> list[Hyperedge]:
        rows = self._conn.execute("SELECT * FROM hyperedges ORDER BY created_at").fetchall()
        return [self._row_to_edge(r) for r in rows]

    def edges_for_node(self, node_id: str) -> list[Hyperedge]:
        rows = self._conn.execute(
            """SELECT h.* FROM hyperedges h
               JOIN hyperedge_nodes hn ON hn.hyperedge_id = h.id
               WHERE hn.node_id = ?""",
            (node_id,),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def edges_of_type(self, edge_type: str) -> list[Hyperedge]:
        rows = self._conn.execute(
            "SELECT * FROM hyperedges WHERE type = ? ORDER BY created_at", (edge_type,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # -- activation: decay + reinforcement --------------------------------
    def live_activation(self, node_id: str, *, curve: DecayFn = exponential_decay) -> float:
        row = self._conn.execute(
            """SELECT a.base, a.updated_at, n.decay_rate
               FROM activations a JOIN nodes n ON n.id = a.node_id
               WHERE a.node_id = ?""",
            (node_id,),
        ).fetchone()
        if row is None:
            return 0.0
        return current_activation(row["base"], now_ts() - row["updated_at"], row["decay_rate"], curve)

    def reinforce_node(
        self,
        node_id: str,
        amount: float = 0.5,
        *,
        curve: DecayFn = exponential_decay,
    ) -> float:
        """Apply Hebbian reinforcement, persisting the new decayed-then-boosted base."""
        live = self.live_activation(node_id, curve=curve)
        boosted = reinforce(live, amount=amount)
        self._conn.execute(
            """UPDATE activations
               SET base = ?, updated_at = ?, access_count = access_count + 1
               WHERE node_id = ?""",
            (boosted, now_ts(), node_id),
        )
        self._conn.commit()
        return boosted

    def set_activation(self, node_id: str, base: float, updated_at: float | None = None) -> None:
        self._conn.execute(
            "UPDATE activations SET base = ?, updated_at = ? WHERE node_id = ?",
            (base, updated_at if updated_at is not None else now_ts(), node_id),
        )
        self._conn.commit()

    def access_count(self, node_id: str) -> int:
        row = self._conn.execute(
            "SELECT access_count FROM activations WHERE node_id = ?", (node_id,)
        ).fetchone()
        return int(row["access_count"]) if row else 0

    # -- search: FTS5 + embeddings ----------------------------------------
    def fts_search(self, query: str, limit: int = 10) -> list[str]:
        """Lexical seed discovery. Returns node ids best-matching the query."""
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                """SELECT n.id FROM nodes_fts f
                   JOIN nodes n ON n.rowid = f.rowid
                   WHERE nodes_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["id"] for r in rows]

    def embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (matrix, ids) of all stored embeddings for numpy search."""
        rows = self._conn.execute("SELECT node_id, vector FROM embeddings").fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        ids = [r["node_id"] for r in rows]
        mat = np.vstack([emb.from_blob(r["vector"]) for r in rows])
        return mat.astype(np.float32), ids

    def semantic_search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        mat, ids = self.embedding_matrix()
        return emb.cosine_search(query_vec, mat, ids, top_k=top_k)

    # -- misc --------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        n = self._conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        e = self._conn.execute("SELECT COUNT(*) c FROM hyperedges").fetchone()["c"]
        m = self._conn.execute("SELECT COUNT(*) c FROM hyperedge_nodes").fetchone()["c"]
        return {"nodes": n, "hyperedges": e, "members": m}

    def raw(self) -> sqlite3.Connection:
        return self._conn


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query over its alnum tokens."""
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
