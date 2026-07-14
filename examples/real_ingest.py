"""End-to-end demo: LLM-based ingestion into a MeshMind hypergraph.

Takes a paragraph, extracts a typed N-ary hyperedge (with participant roles and
a timestamp) via Bedrock Claude Opus, and shows how it lands as a MeshMind
hyperedge you can recall.

Run it::

    python examples/real_ingest.py

If a Bedrock key (``AWS_BEARER_TOKEN_BEDROCK``) is present and ``boto3`` is
installed, it calls the real model. Otherwise it transparently falls back to a
deterministic mock so the demo always runs.
"""

from __future__ import annotations

from meshmind import Mesh
from meshmind.ingest.extractor import bedrock_available, choose_extractor

PARAGRAPH = (
    "On July 13, Eli and David discussed the TEDx application and decided to "
    "focus on the ShapeForge -> Amazon narrative."
)

# A canned response used only if we can't reach Bedrock, so the demo is always
# runnable and shows the exact same shape a real extraction produces.
MOCK_RESPONSE = {
    "entities": [
        {"name": "Eli", "kind": "person", "confidence": 0.98},
        {"name": "David", "kind": "person", "confidence": 0.98},
        {"name": "TEDx application", "kind": "project", "confidence": 0.9},
        {"name": "ShapeForge", "kind": "project", "confidence": 0.88},
        {"name": "Amazon", "kind": "concept", "confidence": 0.8},
    ],
    "hyperedge": {
        "type": "Decision",
        "participants": [
            {"entity_name": "Eli", "role": "decider", "weight": 1.0},
            {"entity_name": "David", "role": "decider", "weight": 1.0},
            {"entity_name": "TEDx application", "role": "topic", "weight": 0.7},
            {"entity_name": "ShapeForge", "role": "subject", "weight": 0.7},
            {"entity_name": "Amazon", "role": "goal", "weight": 0.6},
        ],
        "timestamp": "2026-07-13",
        "confidence": 0.93,
        "provenance": {"source_text": PARAGRAPH},
    },
    "contradictions": [],
    "supersedes": [],
}


def main() -> None:
    live = bedrock_available()
    print("=" * 72)
    print("MeshMind — real LLM ingestion demo")
    print(f"Bedrock reachable: {live}  (model: global.anthropic.claude-opus-4-8)")
    print("=" * 72)
    print(f"\nInput paragraph:\n  {PARAGRAPH}\n")

    if live:
        extractor = choose_extractor(use_llm=True)
    else:
        print("(no Bedrock key found — using deterministic mock response)\n")
        extractor = choose_extractor(mock_mode=True)
        extractor.mock_response = MOCK_RESPONSE

    memory = extractor.extract(PARAGRAPH)

    print("Extracted entities:")
    for e in memory.entities:
        print(f"  - {e.name:<18} kind={e.kind:<8} confidence={e.confidence:.2f}")

    edge = memory.hyperedge
    print(f"\nExtracted hyperedge:  type={edge.type}  confidence={edge.confidence:.2f}")
    print(f"  timestamp: {edge.timestamp}")
    print("  participants (role @ weight):")
    for p in edge.participants:
        print(f"    - {p.entity_name:<18} {p.role:<10} @ {p.weight:.2f}")

    # --- now show it becoming a real MeshMind hyperedge ---------------------
    mesh = Mesh(":memory:")
    mesh.ingest_text(PARAGRAPH, extractor=extractor)

    print("\nPersisted as a MeshMind hypergraph fragment:")
    stats = mesh.stats()
    print(f"  nodes={stats['nodes']}  hyperedges={stats['hyperedges']}")

    stored = mesh.store.edges_of_type(edge.type)[0]
    print(f"\n  Hyperedge {stored.id}  ({stored.type}, arity {stored.arity}):")
    for m in stored.members:
        node = mesh.store.get_node(m.node_id)
        label = node.text if node else m.node_id
        print(f"    [{m.role:<10}] {label}")
    if stored.metadata.get("timestamp"):
        print(f"  when: {stored.metadata['timestamp']}")

    print("\nRecall 'ShapeForge':")
    result = mesh.recall("ShapeForge")
    print("  " + result.to_context_string().replace("\n", "\n  "))

    mesh.close()
    print("\nDone. One episode → one N-ary hyperedge, roles and time intact.")


if __name__ == "__main__":
    main()
