"""MeshMind hello world: ingest -> recall -> inspect, in under 40 lines."""

from meshmind import Mesh

mesh = Mesh(":memory:")  # in-memory hypergraph

# Remember a few related facts. A two-participant turn produces an Experience
# hyperedge of arity >= 3 (statement + Eli + topic node).
eli = mesh.remember("Eli is building MeshMind", participants=["Eli"], context={"topic": "MeshMind"})
mesh.remember("MeshMind uses hypergraphs", context={"topic": "MeshMind"})
mesh.remember("Hypergraphs beat knowledge graphs for memory", context={"topic": "MeshMind"})

# Show a real hyperedge with arity >= 3.
edge = max(mesh.store.all_hyperedges(), key=lambda e: e.arity)
print(f"Widest hyperedge: {edge.type} with arity {edge.arity}")
print("  members:", [(m.role, m.node_id[:10]) for m in edge.members])
print()

# Recall a connected subgraph, not a flat list of chunks.
result = mesh.recall("what is meshmind", budget_tokens=300)
print(result.to_markdown())
print()

# Fresh vs decayed activation: reinforce one node, backdate another.
import time
from meshmind.decay import SECONDS_PER_DAY

mesh.store.reinforce_node(eli.id)  # Hebbian boost from access
fresh = mesh.store.live_activation(eli.id)

other = mesh.remember("An old, fading memory", context={"topic": "old"})
mesh.store.set_activation(other.id, base=1.0, updated_at=time.time() - 60 * SECONDS_PER_DAY)
decayed = mesh.store.live_activation(other.id)

print(f"Fresh (reinforced) activation: {fresh:.3f}")
print(f"Decayed (60 days old)  activation: {decayed:.3f}")
