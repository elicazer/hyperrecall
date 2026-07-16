"""Phase 1 (v2): ingest conv-26 through the dense, coreference-aware extractor.

The v1 runner (``phase1_ingest.py``) asks Gemini for one hyperedge per turn.
This runner points the pipeline at :class:`meshmind.ingest.extractor_v2.ExtractorV2`
instead — the 3-pass extractor that emits many typed N-ary hyperedges per turn
and canonicalizes entities across turns.

It re-ingests conv-26 (or any conv) into a real Mesh via
``mesh.ingest_text(extractor=ext_v2)`` and reports graph shape + coreference
stats. Use ``--limit N`` to prove the pipeline end-to-end on a slice without a
full (paid, ~419-turn) run; ``--mock`` to run offline.

    python phase1_ingest_v2.py --limit 8
    python phase1_ingest_v2.py --mock --limit 20
    python phase1_ingest_v2.py                 # full conv-26

Output: benchmarks/locomo/runs/phase1_v2/<conv>.sqlite  (Mesh)
        benchmarks/locomo/runs/phase1_v2/<conv>.stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # meshmind
sys.path.insert(0, str(ROOT))  # harness package

# Best-effort load of the local Gemini key file (same convention as v1 runner).
env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from harness.load import load  # noqa: E402

from meshmind import Mesh  # noqa: E402
from meshmind.ingest.extractor_v2 import ExtractorV2  # noqa: E402

OUT_DIR = ROOT / "runs" / "phase1_v2"


def graph_stats(mesh: Mesh) -> dict:
    c = mesh.store._conn
    n_nodes = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    n_edges = c.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0]
    n_members = c.execute("SELECT COUNT(*) FROM hyperedge_nodes").fetchone()[0]
    edge_types = c.execute(
        "SELECT type, COUNT(*) FROM hyperedges GROUP BY type ORDER BY 2 DESC"
    ).fetchall()
    node_kinds = c.execute(
        "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY 2 DESC"
    ).fetchall()
    # Coreference gain: distinct entity nodes vs distinct normalized names.
    entity_kinds = ("person", "project", "decision", "event", "place", "time",
                    "artifact", "belief", "preference", "entity")
    ph = ",".join("?" for _ in entity_kinds)
    ent_rows = c.execute(
        f"SELECT lower(text) FROM nodes WHERE kind IN ({ph})", entity_kinds
    ).fetchall()
    distinct_names = len({r[0] for r in ent_rows})
    return {
        "nodes": n_nodes,
        "hyperedges": n_edges,
        "members": n_members,
        "avg_arity": round(n_members / n_edges, 2) if n_edges else 0,
        "entity_nodes": len(ent_rows),
        "distinct_entity_names": distinct_names,
        "edge_types": {t: n for t, n in edge_types},
        "node_kinds": {k: n for k, n in node_kinds},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--limit", type=int, default=0, help="max turns (0 = all)")
    ap.add_argument("--mock", action="store_true", help="offline heuristic, no API")
    args = ap.parse_args(argv)

    mock = args.mock or not os.environ.get("GEMINI_API_KEY")
    if not mock and not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set (use --mock for offline)", file=sys.stderr)
        return 2

    conv = next(c for c in load() if c.sample_id == args.conv)
    turns = [(s, t) for s in conv.sessions for t in s.turns]
    if args.limit:
        turns = turns[: args.limit]
    print(f"{conv.sample_id}: ingesting {len(turns)} turns via extractor v2 "
          f"(mock={mock})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUT_DIR / f"{conv.sample_id}.sqlite"
    if db_path.exists():
        db_path.unlink()

    mesh = Mesh(str(db_path))
    ext = ExtractorV2(mock_mode=mock, logger=lambda m: None)  # quiet per-merge logs

    t0 = time.time()
    n_ok = n_fail = total_edges = total_merges = 0
    for sess, turn in turns:
        try:
            te = mesh.ingest_text(
                turn.text,
                extractor=ext,
                speaker=turn.speaker,
                when=sess.date_time,
                provenance={"dia_id": turn.dia_id, "session": sess.index,
                            "speaker": turn.speaker},
            )
            n_ok += 1
            total_edges += len(te.edge_ids)
            total_merges += len(te.merges)
        except Exception as e:  # noqa: BLE001 - report and continue
            n_fail += 1
            print(f"  ✗ [{turn.dia_id}] {e}", flush=True)
        if (n_ok + n_fail) % 10 == 0:
            el = time.time() - t0
            print(f"  {n_ok+n_fail}/{len(turns)} ok={n_ok} fail={n_fail} "
                  f"edges={total_edges} merges={total_merges} "
                  f"{(n_ok+n_fail)/el:.2f} turns/s", flush=True)

    stats = graph_stats(mesh)
    out = {
        "conv": conv.sample_id,
        "turns_ingested": len(turns),
        "ok": n_ok,
        "fail": n_fail,
        "elapsed_s": round(time.time() - t0, 1),
        "total_hyperedges": total_edges,
        "total_merges": total_merges,
        "edges_per_turn": round(total_edges / n_ok, 2) if n_ok else 0,
        "graph": stats,
        "mock": mock,
    }
    (OUT_DIR / f"{conv.sample_id}.stats.json").write_text(json.dumps(out, indent=2))
    print("\n=== RESULT ===")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
