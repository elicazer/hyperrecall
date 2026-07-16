"""Mem0 Cloud adapter with an explicit credential boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness.load import Conversation, iter_turns
from harness.systems.base import Retrieval, System


class Mem0System(System):
    name = "mem0"

    def __init__(self, locomo_root: Path, conv_id: str) -> None:
        key = os.environ.get("MEM0_API_KEY")
        if not key:
            raise RuntimeError("Set MEM0_API_KEY / ZEP_API_KEY to enable this system")
        try:
            from mem0 import MemoryClient
        except ImportError as exc:
            raise RuntimeError("Install mem0ai to enable Mem0 Cloud") from exc
        self.client = MemoryClient(api_key=key)
        self.user_id = f"locomo-{conv_id}"
        self.ingested = 0

    def ingest(self, conversation: Conversation) -> None:
        for session, turn in iter_turns(conversation):
            self.client.add(
                f"[{session.date_time}] {turn.speaker}: {turn.text}",
                user_id=self.user_id,
                metadata={"dia_id": turn.dia_id, "session": session.index},
            )
            self.ingested += 1

    def retrieve(self, question: str) -> Retrieval:
        response = self.client.search(question, user_id=self.user_id, top_k=10)
        hits = response.get("results", response) if isinstance(response, dict) else response
        lines = [(hit.get("memory") or hit.get("text") or "") for hit in hits]
        lines = [line for line in lines if line]
        return Retrieval("\n".join(lines) or "(no relevant memory found)", {"n_memories": len(lines)})

    def close(self) -> None:
        return None

    def cost_record(self) -> dict[str, Any]:
        return {"usd": None, "turns": self.ingested, "note": "Mem0 Cloud usage is provider-metered"}
