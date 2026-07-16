"""Zep Cloud adapter.

Zep's current Python product requires a cloud API key.  We deliberately do not
silently create an account or substitute a different product for the benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness.load import Conversation
from systems.base import Retrieval


class ZepSystem:
    name = "zep"

    def __init__(self, run_root: Path, conv_id: str) -> None:
        if not os.environ.get("ZEP_API_KEY"):
            raise RuntimeError(
                "Zep requires ZEP_API_KEY for zep-cloud. Create a free-tier key at "
                "https://app.getzep.com only after user approval, install zep-cloud, "
                "then rerun. The legacy self-hosted zep-python API is not equivalent "
                "to the current Zep Context Graph benchmark product."
            )
        raise RuntimeError(
            "ZEP_API_KEY is present but this reproducibility build intentionally "
            "requires a pinned, validated Zep Cloud adapter before paid execution."
        )

    def ingest(self, conversation: Conversation) -> None:
        raise AssertionError("unreachable")

    def retrieve(self, question: str) -> Retrieval:
        raise AssertionError("unreachable")

    def close(self) -> None:
        return None

    def cost_record(self) -> dict[str, Any]:
        return {"usd": 0.0, "note": "stub"}
