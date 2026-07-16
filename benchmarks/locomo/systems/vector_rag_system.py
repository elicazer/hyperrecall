"""Raw-turn MiniLM vector-RAG baseline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from harness.load import Conversation
from phase2_retrieval import EMBEDDER_NAME, VectorRag
from systems.base import Retrieval


class VectorRagSystem:
    name = "vector_rag"

    def __init__(self, run_root: Path, conv_id: str) -> None:
        self.embedder = SentenceTransformer(EMBEDDER_NAME)
        self.index: VectorRag | None = None

    def ingest(self, conversation: Conversation) -> None:
        self.index = VectorRag(conversation, self.embedder)

    def retrieve(self, question: str) -> Retrieval:
        if self.index is None:
            raise RuntimeError("vector-RAG conversation has not been ingested")
        context, metadata = self.index.retrieve(question)
        return Retrieval(context, metadata)

    def close(self) -> None:
        return None

    def cost_record(self) -> dict[str, Any]:
        return {"usd": 0.0, "note": "local MiniLM embeddings"}
