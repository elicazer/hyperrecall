"""Smoke test for MeshMind's LLM ingest — Gemini flavor.

Not part of the LoCoMo benchmark; this just answers "does an LLM produce a
plausible MeshMind-shaped extraction for a handful of turns?" before we spend
real money running the full pipeline on 5,882 turns.

Uses Gemini 2.0 Flash as the extractor. Mirrors the same JSON shape that
PR #1's Bedrock `record_memory` tool enforces (see
``meshmind.ingest.extractor.RECORD_MEMORY_TOOL``), so a pass here is
architecturally meaningful even though the LoCoMo run will use Haiku/Bedrock.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Load Gemini key
env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path.home() / "projects" / "meshmind" / "src"))

from google import genai
from google.genai import types

from meshmind.ingest.extractor import RECORD_MEMORY_TOOL, _SYSTEM_PROMPT

MODEL = "gemini-2.5-flash"

# --------------------------------------------------------------------------- #
# 5 hand-crafted turns — a mini "does memory work" gauntlet.
# --------------------------------------------------------------------------- #
TURNS: list[dict] = [
    {
        "id": "T1_single_fact",
        "speaker": "Eli",
        "text": "I just adopted a puppy named Milo, he's a border collie.",
    },
    {
        "id": "T2_state_change",
        "speaker": "Eli",
        "text": (
            "Actually Milo isn't a border collie — the shelter got it wrong. "
            "The vet confirmed he's an Australian shepherd mix."
        ),
    },
    {
        "id": "T3_multi_entity_event",
        "speaker": "Eli",
        "text": (
            "Sarah and I took Milo hiking on the Bommer Canyon trail on Saturday. "
            "Ran into her brother James halfway through."
        ),
    },
    {
        "id": "T4_temporal",
        "speaker": "Eli",
        "text": (
            "My animatronic head demo is scheduled for the Bay Area Maker Faire "
            "on October 19, 2026 at 10am."
        ),
    },
    {
        "id": "T5_preference",
        "speaker": "Eli",
        "text": "I prefer Fusion 360 over Onshape for hardware CAD work.",
    },
]

# --------------------------------------------------------------------------- #
# Gemini call — mirror the record_memory JSON schema.
# --------------------------------------------------------------------------- #
def gemini_schema_from_tool() -> dict:
    """Adapt PR #1's JSON-Schema to Gemini's response_schema.

    Gemini rejects array-of-types (``["string", "null"]``); use ``nullable: true``.
    """
    import copy

    schema = copy.deepcopy(RECORD_MEMORY_TOOL["inputSchema"]["json"])

    def _fix(node):
        if isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x != "null"]
                node["type"] = non_null[0] if non_null else "string"
                if "null" in t:
                    node["nullable"] = True
            for v in node.values():
                _fix(v)
        elif isinstance(node, list):
            for v in node:
                _fix(v)

    _fix(schema)
    return schema


def extract_via_gemini(client: genai.Client, text: str) -> dict:
    """Call Gemini with force-JSON via response_schema. Returns the parsed dict."""
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=gemini_schema_from_tool(),
            temperature=0.2,
        ),
    )
    return json.loads(resp.text)


# --------------------------------------------------------------------------- #
# Cheap eyeball checks — not scored, just flagged.
# --------------------------------------------------------------------------- #
def check(payload: dict, turn: dict) -> list[str]:
    issues: list[str] = []
    if "entities" not in payload or not payload["entities"]:
        issues.append("no entities")
    if "hyperedge" not in payload:
        issues.append("no hyperedge")
        return issues
    edge = payload["hyperedge"]
    if not edge.get("type"):
        issues.append("hyperedge has no type")
    if not edge.get("participants"):
        issues.append("hyperedge has no participants")

    # entity-set sanity checks per turn
    ents_lower = " ".join(e.get("name", "").lower() for e in payload["entities"])
    if turn["id"] == "T1_single_fact":
        if "milo" not in ents_lower:
            issues.append("missing Milo")
    elif turn["id"] == "T2_state_change":
        if "milo" not in ents_lower:
            issues.append("missing Milo")
        # ideally supersedes/contradictions non-empty
        if not payload.get("contradictions") and not payload.get("supersedes"):
            issues.append("no contradiction/supersession on breed correction")
    elif turn["id"] == "T3_multi_entity_event":
        for name in ("milo", "sarah", "james"):
            if name not in ents_lower:
                issues.append(f"missing {name}")
    elif turn["id"] == "T4_temporal":
        ts = edge.get("timestamp")
        if not ts or "2026" not in str(ts):
            issues.append(f"missing/bad timestamp: {ts!r}")
    elif turn["id"] == "T5_preference":
        for name in ("fusion 360", "onshape"):
            if name not in ents_lower:
                issues.append(f"missing {name}")
    return issues


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    total_issues = 0
    for turn in TURNS:
        print(f"\n─── {turn['id']}  |  {turn['speaker']}: {turn['text']}")
        try:
            payload = extract_via_gemini(client, turn["text"])
        except Exception as e:
            print(f"  ✗ Gemini error: {e}")
            total_issues += 1
            continue
        print("  extraction:")
        print("   ", json.dumps(payload, indent=2).replace("\n", "\n    "))
        issues = check(payload, turn)
        if issues:
            print("  ⚠️  issues:", ", ".join(issues))
            total_issues += len(issues)
        else:
            print("  ✅ looks good")

    print(f"\nTotal issues across {len(TURNS)} turns: {total_issues}")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
