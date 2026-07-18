"""Contradiction detection: HyperRecall surfaces both sides with a conflict flag."""

from hyperrecall import Mesh

mesh = Mesh(":memory:")

# Two memories that cannot both be true.
a = mesh.remember("The TEDx event will be held in Newport Beach", context={"topic": "TEDx"})
b = mesh.remember("The TEDx event will be held in Irvine", context={"topic": "TEDx"})

# Explicitly link them as contradictory. (Later, an LLM ingest step does this
# automatically; for now we assert it.)
mesh.contradict(a.id, b.id, note="Two different venues claimed for the same event")

print("Known contradictions:")
for node_a, node_b, edge in mesh.contradictions():
    print(f"  - {node_a.text!r}  <>  {node_b.text!r}")
    print(f"    note: {edge.metadata.get('note')}")
print()

# Recall surfaces BOTH sides, each flagged, so the agent can reason about the
# conflict rather than silently picking one.
result = mesh.recall("Where is the TEDx event?")
print(result.to_markdown())
print()
print("Prompt-ready context (conflicts inlined):")
print(result.to_context_string())
