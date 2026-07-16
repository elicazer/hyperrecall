"""MeshMind adapter, preserving the phase-2 retrieval configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from meshmind import Mesh
from harness.load import Conversation
from phase2_retrieval import EMBEDDER_NAME, render_mesh_context
from systems.base import Retrieval


class MeshMindSystem:
    name = "meshmind"

    def __init__(self, run_root: Path, conv_id: str) -> None:
        phase1 = run_root.parent / "phase1"
        embedded = phase1 / f"{conv_id}.embed.sqlite"
        self.db_path = embedded if embedded.exists() else phase1 / f"{conv_id}.sqlite"
        if not self.db_path.exists():
            raise RuntimeError(
                f"pre-built MeshMind database missing: {self.db_path}; run phase1_ingest.py"
            )
        embedder = SentenceTransformer(EMBEDDER_NAME)

        def embed(text: str) -> np.ndarray:
            return embedder.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)

        self.mesh = Mesh(str(self.db_path), embed=embed)

    def ingest(self, conversation: Conversation) -> None:
        # The expensive extraction is a benchmark input artifact, built by phase 1.
        return None

    def retrieve(self, question: str) -> Retrieval:
        context, metadata = render_mesh_context(self.mesh, question)
        metadata["db_path"] = str(self.db_path)
        return Retrieval(context, metadata)

    def close(self) -> None:
        close = getattr(getattr(self.mesh, "store", None), "close", None)
        if close:
            close()

    def cost_record(self) -> dict[str, Any]:
        return {"usd": 0.0, "note": "pre-built phase1 ingestion excluded"}
