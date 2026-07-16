"""Common contract for systems compared by the LoCoMo harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.load import Conversation


@dataclass
class Retrieval:
    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


class System(ABC):
    """A memory backend. Answer generation and judging remain harness-owned."""

    name: str

    @abstractmethod
    def ingest(self, conversation: Conversation) -> None:
        """Make one normalized conversation available for retrieval."""

    @abstractmethod
    def retrieve(self, question: str) -> Retrieval:
        """Return evidence context, never a final answer."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the adapter."""

    @abstractmethod
    def cost_record(self) -> dict[str, Any]:
        """Return measured or clearly identified estimated system cost."""
