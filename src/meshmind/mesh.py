"""The public MeshMind API.

``Mesh`` is the one class most users touch. It wires the storage, ingestion and
retrieval layers together behind an ergonomic surface::

    from meshmind import Mesh

    mesh = Mesh(":memory:")
    mesh.remember("Eli is building MeshMind", participants=["Eli"])
    result = mesh.recall("what is Eli building?")
    print(result.to_context_string())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .decay import DecayFn, exponential_decay, get_curve
from .ingest.extractor import (
    ExtractedMemory,
    Extraction,
    HeuristicExtractor,
    LLMExtractor,
    choose_extractor,
    extract,
)
from .models import (
    CONTRADICTS,
    SUPERSEDES,
    Hyperedge,
    HyperedgeMember,
    Node,
)
from .retrieval.query import Subgraph, recall
from .storage.embeddings import DEFAULT_DIM, EmbedFn, hash_embed
from .storage.sqlite_store import SqliteStore


class Mesh:
    """A hypergraph memory for LLM agents."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        embed: EmbedFn | None = None,
        decay_curve: str | DecayFn = "exponential",
    ) -> None:
        self.store = SqliteStore(path)
        self.embed: EmbedFn = embed or (lambda t: hash_embed(t, DEFAULT_DIM))
        self.curve: DecayFn = decay_curve if callable(decay_curve) else get_curve(decay_curve)

    # -- write path --------------------------------------------------------
    def remember(
        self,
        text: str,
        *,
        participants: list[str] | None = None,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
    ) -> Node:
        """Ingest an utterance. Returns the primary node created.

        The utterance is decomposed into a small hypergraph fragment (a
        statement node, participant/context nodes, and an ``Experience`` edge
        binding them) and persisted.
        """
        ex = extract(
            text,
            participants=participants,
            context=context,
            confidence=confidence,
            provenance=provenance,
        )
        self._persist(ex)
        assert ex.primary is not None
        return ex.primary

    def ingest_text(
        self,
        text: str,
        *,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
        use_llm: bool | None = None,
        mock_mode: bool = False,
        extractor: LLMExtractor | HeuristicExtractor | str | None = None,
        speaker: str | None = None,
        when: str | None = None,
    ) -> ExtractedMemory | Any:
        """Ingest raw natural-language text with the LLM extractor.

        Unlike :meth:`remember` (which expects you to name participants), this
        uses the real LLM pipeline (:class:`LLMExtractor`) to decompose the text
        into entities and one typed N-ary hyperedge, then persists them. When no
        Bedrock key is configured it transparently falls back to the
        deterministic :class:`HeuristicExtractor`.

        Pass ``extractor="v2"`` (or an :class:`~meshmind.ingest.extractor_v2.ExtractorV2`
        instance) to use the dense, coreference-aware 3-pass pipeline; it
        canonicalizes entities across turns and returns a ``TurnExtraction``.

        Pass ``mock_mode=True`` (or inject ``extractor``) to run offline. Returns
        the validated :class:`ExtractedMemory`; the created nodes and hyperedge
        are already persisted and recallable.
        """
        # -- v2 pipeline (dense + coreference-aware) ------------------------
        if extractor == "v2" or getattr(extractor, "backend", None) == "v2":
            from .ingest.extractor_v2 import ExtractorV2

            ev2 = extractor if isinstance(extractor, ExtractorV2) else ExtractorV2(mock_mode=mock_mode)
            return ev2.ingest(
                self,
                text,
                speaker=speaker,
                when=when,
                context=context,
                confidence=confidence,
                provenance=provenance,
            )

        ext = extractor or choose_extractor(use_llm=use_llm, mock_mode=mock_mode)
        memory = ext.extract(
            text,
            context=context,
            confidence=confidence,
            provenance=provenance,
        )
        ex = memory.to_extraction(
            confidence=confidence if confidence != 1.0 else None,
            context=context,
            provenance=provenance,
        )
        self._persist(ex)
        return memory

    def add_node(self, node: Node) -> Node:
        """Add a pre-built node (embeds its text automatically)."""
        return self.store.add_node(node, vector=self.embed(node.text))

    def add_hyperedge(self, edge: Hyperedge) -> Hyperedge:
        """Add a pre-built hyperedge (must have arity >= 2)."""
        return self.store.add_hyperedge(edge)

    def _persist(self, ex: Extraction) -> None:
        for node in ex.nodes:
            self.store.add_node(node, vector=self.embed(node.text))
        for edge in ex.hyperedges:
            self.store.add_hyperedge(edge)

    # -- relation helpers --------------------------------------------------
    def contradict(self, node_a: str, node_b: str, *, note: str | None = None) -> Hyperedge:
        """Record that two nodes contradict each other."""
        edge = Hyperedge(
            type=CONTRADICTS,
            members=[
                HyperedgeMember(node_a, role="claim"),
                HyperedgeMember(node_b, role="claim"),
            ],
            metadata={"note": note} if note else {},
        )
        return self.store.add_hyperedge(edge)

    def supersede(self, old_id: str, new_id: str, *, note: str | None = None) -> Hyperedge:
        """Record that ``new_id`` supersedes ``old_id``."""
        edge = Hyperedge(
            type=SUPERSEDES,
            members=[
                HyperedgeMember(old_id, role="old", weight=0.5),
                HyperedgeMember(new_id, role="new", weight=1.0),
            ],
            metadata={"note": note} if note else {},
        )
        return self.store.add_hyperedge(edge)

    # -- read path ---------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        budget_tokens: int | None = None,
        k_hops: int = 2,
        max_seeds: int = 5,
        prefer_newest: bool = True,
        reinforce_on_access: bool = True,
        sim_rerank: float = 0.0,
    ) -> Subgraph:
        """Retrieve a connected subgraph of memories relevant to ``query``."""
        return recall(
            self.store,
            query,
            embed=self.embed,
            budget_tokens=budget_tokens,
            k_hops=k_hops,
            max_seeds=max_seeds,
            prefer_newest=prefer_newest,
            reinforce_on_access=reinforce_on_access,
            sim_rerank=sim_rerank,
            curve=self.curve,
        )

    # -- introspection -----------------------------------------------------
    def inspect_node(self, node_id: str) -> dict[str, Any]:
        """Return everything known about a node: content, live activation, edges."""
        node = self.store.get_node(node_id, curve=self.curve)
        if node is None:
            raise KeyError(node_id)
        edges = self.store.edges_for_node(node_id)
        return {
            "id": node.id,
            "text": node.text,
            "kind": node.kind,
            "confidence": node.confidence,
            "activation": self.store.live_activation(node_id, curve=self.curve),
            "access_count": self.store.access_count(node_id),
            "created_at": node.created_at,
            "metadata": node.metadata,
            "edges": [
                {"id": e.id, "type": e.type, "arity": e.arity, "role": e.role_of(node_id)}
                for e in edges
            ],
        }

    def contradictions(self) -> list[tuple[Node, Node, Hyperedge]]:
        """Return every pair of contradicting nodes with the edge that links them."""
        pairs: list[tuple[Node, Node, Hyperedge]] = []
        for edge in self.store.edges_of_type(CONTRADICTS):
            ids = edge.node_ids
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a = self.store.get_node(ids[i], curve=self.curve)
                    b = self.store.get_node(ids[j], curve=self.curve)
                    if a and b:
                        pairs.append((a, b, edge))
        return pairs

    def stats(self) -> dict[str, int]:
        return self.store.counts()

    # -- portability -------------------------------------------------------
    def export(self, directory: str | Path) -> Path:
        """Export the whole mesh to a portable markdown+YAML directory."""
        from .portable.exporter import export_dir

        return export_dir(self.store, directory)

    @classmethod
    def import_dir(
        cls,
        directory: str | Path,
        path: str | Path = ":memory:",
        *,
        embed: EmbedFn | None = None,
    ) -> "Mesh":
        """Build a mesh from a previously-exported portable directory."""
        from .portable.importer import import_dir

        mesh = cls(path, embed=embed)
        import_dir(mesh.store, directory, embed=mesh.embed)
        return mesh

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Mesh":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
