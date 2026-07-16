"""Local Mem0 OSS adapter using its native OpenAI-backed extraction pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harness.load import Conversation, iter_turns
from systems.base import Retrieval


class Mem0System:
    name = "mem0"

    def __init__(self, run_root: Path, conv_id: str) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("local Mem0 requires OPENAI_API_KEY")
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError("install the local SDK with: pip install mem0ai") from exc

        self.state_dir = run_root / "mem0_state" / conv_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.marker = self.state_dir / "ingest.complete.json"
        self.user_id = f"locomo-{conv_id}"
        self.turn_count = 0
        self.input_chars = 0
        # Pinning the model is essential for reproducibility. gpt-4o-mini keeps
        # the pilot economical and is still Mem0's own extraction/update prompt.
        config = {
            "version": "v1.1",
            "llm": {"provider": "openai", "config": {
                "model": os.environ.get("MEM0_LLM_MODEL", "gpt-4o-mini"),
                "temperature": 0.0,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
            }},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"locomo_{conv_id.replace('-', '_')}",
                "path": str(self.state_dir / "qdrant"),
                "on_disk": True,
            }},
            "history_db_path": str(self.state_dir / "history.sqlite"),
        }
        self.llm_model = config["llm"]["config"]["model"]
        self.memory = Memory.from_config(config)

    def ingest(self, conversation: Conversation) -> None:
        if self.marker.exists():
            data = json.loads(self.marker.read_text())
            self.turn_count = data["turn_count"]
            self.input_chars = data["input_chars"]
            return
        for index, (session, turn) in enumerate(iter_turns(conversation), 1):
            role = "user" if turn.speaker == conversation.speaker_a else "assistant"
            content = f"[{session.date_time}] {turn.speaker}: {turn.text}"
            self.memory.add(
                [{"role": role, "content": content}],
                user_id=self.user_id,
                metadata={"dia_id": turn.dia_id, "session": session.index},
            )
            self.turn_count += 1
            self.input_chars += len(content)
            if index % 25 == 0:
                print(f"  [mem0] ingested {index} turns", flush=True)
        self.marker.write_text(json.dumps({
            "turn_count": self.turn_count, "input_chars": self.input_chars,
            "llm_model": self.llm_model, "mem0_mode": "local OSS",
        }, indent=2) + "\n")

    def retrieve(self, question: str) -> Retrieval:
        response = self.memory.search(
            question, filters={"user_id": self.user_id}, top_k=10, threshold=0.0,
        )
        hits = response.get("results", response if isinstance(response, list) else [])
        lines = []
        for hit in hits:
            memory = hit.get("memory") or hit.get("text") or ""
            if memory:
                lines.append(memory)
        context = "\n".join(lines) if lines else "(no relevant memory found)"
        return Retrieval(context, {"n_memories": len(lines), "top_k": 10})

    def close(self) -> None:
        client = getattr(getattr(self.memory, "vector_store", None), "client", None)
        if client and hasattr(client, "close"):
            client.close()

    def cost_record(self) -> dict[str, Any]:
        # Mem0 does not expose nested provider usage. Estimate conservatively:
        # prompt overhead + source, 60 extraction output tokens/turn, embeddings.
        source_tokens = (self.input_chars + 3) // 4
        llm_input = source_tokens + self.turn_count * 500
        llm_output = self.turn_count * 60
        llm_usd = (llm_input * 0.15 + llm_output * 0.60) / 1_000_000
        embedding_usd = source_tokens * 0.02 / 1_000_000
        return {
            "usd": llm_usd + embedding_usd, "estimated": True,
            "turns": self.turn_count, "llm_model": self.llm_model,
            "estimated_input_tokens": llm_input,
            "estimated_output_tokens": llm_output,
            "note": "nested Mem0 OpenAI usage is not exposed by mem0ai",
        }
