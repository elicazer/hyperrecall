"""Portable export: a MeshMind database is a directory of Markdown files.

Run this, then open ``mesh_export/nodes/*.md`` — each memory is a human-readable
file with YAML frontmatter. Re-importing reproduces the database losslessly:
the "USB-C for AI memory" angle.
"""

from pathlib import Path

from meshmind import Mesh

mesh = Mesh(":memory:")
mesh.remember(
    "Eli asked David about TEDx applications on July 13",
    participants=["Eli", "David"],
    context={"topic": "TEDx", "session": "abc123"},
    confidence=0.9,
)
mesh.remember("MeshMind exports to portable Markdown", context={"topic": "MeshMind"})

out = Path("mesh_export")
mesh.export(out)
print(f"Exported {mesh.stats()} to {out}/")
print()

# Show one exported node file verbatim.
sample = sorted((out / "nodes").glob("*.md"))[0]
print(f"--- {sample} ---")
print(sample.read_text())

# Round-trip it back into a fresh database.
restored = Mesh.import_dir(out, ":memory:")
print(f"Re-imported: {restored.stats()}")
result = restored.recall("TEDx")
print("Recall after round-trip:")
print(result.to_context_string())
