"""``meshmind`` command-line interface.

Built on Typer. Subcommands::

    meshmind remember "text" --db mesh.db --participant Eli --topic TEDx
    meshmind recall  "query" --db mesh.db --budget 500
    meshmind export  mesh.db ./out
    meshmind import  ./out  mesh.db
    meshmind stats   mesh.db
    meshmind demo
"""

from __future__ import annotations

from pathlib import Path

import typer

from .mesh import Mesh

app = typer.Typer(add_completion=False, help="Hypergraph memory for LLM agents.")


@app.command()
def remember(
    text: str,
    db: str = typer.Option("mesh.db", help="Path to the mesh database."),
    participant: list[str] = typer.Option(None, "--participant", "-p", help="A participant."),
    topic: str = typer.Option(None, help="Topic context tag."),
    confidence: float = typer.Option(1.0, help="Confidence in [0,1]."),
) -> None:
    """Ingest a memory."""
    mesh = Mesh(db)
    ctx = {"topic": topic} if topic else {}
    node = mesh.remember(text, participants=participant or [], context=ctx, confidence=confidence)
    typer.echo(f"remembered {node.id}: {node.text}")
    mesh.close()


@app.command()
def recall(
    query: str,
    db: str = typer.Option("mesh.db", help="Path to the mesh database."),
    budget: int = typer.Option(None, help="Token budget for the result."),
    hops: int = typer.Option(2, help="Spreading-activation hops."),
) -> None:
    """Recall memories relevant to a query."""
    mesh = Mesh(db)
    result = mesh.recall(query, budget_tokens=budget, k_hops=hops)
    typer.echo(result.to_markdown())
    mesh.close()


@app.command("export")
def export_cmd(
    db: str = typer.Argument(..., help="Source mesh database."),
    directory: str = typer.Argument(..., help="Output directory."),
) -> None:
    """Export a mesh to a portable markdown+YAML directory."""
    mesh = Mesh(db)
    out = mesh.export(directory)
    typer.echo(f"exported {mesh.stats()} to {out}")
    mesh.close()


@app.command("import")
def import_cmd(
    directory: str = typer.Argument(..., help="Portable directory to import."),
    db: str = typer.Argument(..., help="Destination mesh database."),
) -> None:
    """Import a portable directory into a (new) mesh database."""
    mesh = Mesh.import_dir(directory, db)
    typer.echo(f"imported into {db}: {mesh.stats()}")
    mesh.close()


@app.command()
def stats(db: str = typer.Argument(..., help="Mesh database.")) -> None:
    """Show node/edge counts."""
    mesh = Mesh(db)
    typer.echo(mesh.stats())
    mesh.close()


@app.command()
def demo() -> None:
    """Run a tiny in-memory demo end to end."""
    mesh = Mesh(":memory:")
    mesh.remember("Eli is building MeshMind", participants=["Eli"], context={"topic": "MeshMind"})
    mesh.remember("MeshMind uses hypergraphs", context={"topic": "MeshMind"})
    mesh.remember("Hypergraphs beat knowledge graphs for memory", context={"topic": "MeshMind"})
    result = mesh.recall("what is meshmind", budget_tokens=200)
    typer.echo(result.to_markdown())
    mesh.close()


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
