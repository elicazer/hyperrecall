"""Local raw-turn MiniLM vector-RAG baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from harness.load import Conversation
from harness.systems.base import Retrieval, System
from phase2_retrieval import EMBEDDER_NAME, VectorRag


class VectorRagSystem(System):
    name = "vector_rag"

    def __init__(self, locomo_root: Path, conv_id: str) -> None:
        self.embedder = SentenceTransformer(EMBEDDER_NAME)
        self.index: VectorRag | None = None

    def ingest(self, conversation: Conversation) -> None:
        self.index = VectorRag(conversation, self.embedder)

    def retrieve(self, question: str) -> Retrieval:
        if self.index is None:
            raise RuntimeError("vector RAG has not ingested a conversation")
        context, metadata = self.index.retrieve(question)
        return Retrieval(context, metadata)

    def close(self) -> None:
        self.index = None

    def cost_record(self) -> dict[str, Any]:
        return {"usd": 0.0, "note": "local MiniLM embeddings"}
