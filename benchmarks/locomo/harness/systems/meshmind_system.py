"""MeshMind adapter using the existing, pre-built LoCoMo mesh."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from harness.load import Conversation
from harness.systems.base import Retrieval, System
from meshmind import Mesh
from phase2_retrieval import EMBEDDER_NAME, render_mesh_context


class MeshMindSystem(System):
    name = "meshmind"

    def __init__(self, locomo_root: Path, conv_id: str) -> None:
        phase1 = locomo_root / "runs" / "phase1"
        embedded = phase1 / f"{conv_id}.embed.sqlite"
        self.db_path = embedded if embedded.exists() else phase1 / f"{conv_id}.sqlite"
        if not self.db_path.exists():
            raise RuntimeError(f"pre-built MeshMind database missing: {self.db_path}")
        embedder = SentenceTransformer(EMBEDDER_NAME)

        def embed(text: str) -> np.ndarray:
            return embedder.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)

        self.mesh = Mesh(str(self.db_path), embed=embed)

    def ingest(self, conversation: Conversation) -> None:
        # Extraction is intentionally not repeated: the checked benchmark input is
        # the pre-built phase-1 mesh. This adapter makes no LLM calls.
        return None

    def retrieve(self, question: str) -> Retrieval:
        context, metadata = render_mesh_context(self.mesh, question)
        return Retrieval(context, {**metadata, "db_path": str(self.db_path)})

    def close(self) -> None:
        close = getattr(getattr(self.mesh, "store", None), "close", None)
        if close:
            close()

    def cost_record(self) -> dict[str, Any]:
        return {"usd": 0.0, "note": "pre-built phase-1 ingestion excluded"}
