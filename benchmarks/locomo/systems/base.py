"""Common contract implemented by every benchmarked memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.load import Conversation


@dataclass
class Retrieval:
    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemorySystem(Protocol):
    """Only retrieval differs between systems; answering is harness-owned."""

    name: str

    def ingest(self, conversation: Conversation) -> None: ...

    def retrieve(self, question: str) -> Retrieval: ...

    def close(self) -> None: ...

    def cost_record(self) -> dict[str, Any]: ...
