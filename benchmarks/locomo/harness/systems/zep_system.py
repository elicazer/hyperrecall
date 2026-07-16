"""Zep Cloud boundary; no substitute is used when credentials are absent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness.load import Conversation
from harness.systems.base import Retrieval, System


class ZepSystem(System):
    name = "zep"

    def __init__(self, locomo_root: Path, conv_id: str) -> None:
        if not os.environ.get("ZEP_API_KEY"):
            raise RuntimeError("Set MEM0_API_KEY / ZEP_API_KEY to enable this system")
        raise RuntimeError(
            "ZEP_API_KEY is set, but the Zep Cloud adapter needs validation against "
            "the account's installed SDK version before external writes"
        )

    def ingest(self, conversation: Conversation) -> None:
        raise AssertionError("unreachable")

    def retrieve(self, question: str) -> Retrieval:
        raise AssertionError("unreachable")

    def close(self) -> None:
        return None

    def cost_record(self) -> dict[str, Any]:
        return {"usd": None, "note": "not run"}
