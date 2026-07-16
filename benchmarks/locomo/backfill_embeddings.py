"""Backfill real semantic embeddings into a phase-1 mesh.

The phase-1 mesh is built with MeshMind's shipped ``hash_embed`` — a
dependency-free *hashing* embedder that is explicitly **not** semantic (see
``storage/embeddings.py``). That handicaps MeshMind's seed discovery: the
spreading-activation retrieval can only start from lexical token overlap, while
the vector-RAG baseline seeds from real ``all-MiniLM-L6-v2`` sentence
embeddings. To compare the *graph* fairly we must seed MeshMind from the same
embedding space as the baseline.

This script does NOT re-run the (expensive) Gemini extraction. It copies the
existing mesh and only rewrites the ``embeddings`` table, re-encoding every
node's text with MiniLM. Schema is unchanged (the ``dim`` column is per-row).

Usage:  python backfill_embeddings.py
Output: runs/phase1/conv-26.embed.sqlite
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from meshmind.storage import embeddings as emb  # noqa: E402

CONV_ID = "conv-26"
SRC = ROOT / "runs" / "phase1" / f"{CONV_ID}.sqlite"
DST = ROOT / "runs" / "phase1" / f"{CONV_ID}.embed.sqlite"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256


def main() -> int:
    if not SRC.exists():
        print(f"source mesh not found: {SRC}", file=sys.stderr)
        return 2
    shutil.copyfile(SRC, DST)
    print(f"copied {SRC.name} -> {DST.name}")

    conn = sqlite3.connect(str(DST))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, text, kind FROM nodes ORDER BY created_at").fetchall()

    # For fact nodes (the original turns) prepend the speaker, mirroring how the
    # vector-RAG baseline embeds "speaker: text". This is not cosmetic: the
    # question "What is Caroline's identity?" has cosine 0.09 against the bare
    # turn but 0.48 against "Caroline: <turn>" — the speaker name is a strong
    # relevance signal, and embedding without it makes seed discovery blind.
    speaker_of: dict[str, str] = {}
    for r in conn.execute(
        """SELECT hn.node_id AS nid, h.provenance AS prov
           FROM hyperedge_nodes hn JOIN hyperedges h ON h.id = hn.hyperedge_id"""
    ).fetchall():
        if r["nid"] in speaker_of:
            continue
        try:
            sp = (json.loads(r["prov"]) or {}).get("speaker", "")
        except Exception:
            sp = ""
        if sp:
            speaker_of[r["nid"]] = sp

    ids = [r["id"] for r in rows]
    texts = []
    for r in rows:
        sp = speaker_of.get(r["id"])
        if r["kind"] == "fact" and sp:
            texts.append(f"{sp}: {r['text']}")
        else:
            texts.append(r["text"])
    print(f"re-embedding {len(ids)} nodes with {EMBEDDER_NAME} "
          f"({sum(1 for r in rows if r['kind']=='fact')} fact nodes speaker-prefixed)")

    model = SentenceTransformer(EMBEDDER_NAME)
    vecs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH
    ).astype(np.float32)
    dim = int(vecs.shape[1])

    for nid, vec in zip(ids, vecs):
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(node_id, dim, vector) VALUES (?, ?, ?)",
            (nid, dim, emb.to_blob(vec)),
        )
    conn.commit()

    got = conn.execute("SELECT dim, COUNT(*) n FROM embeddings GROUP BY dim").fetchall()
    print("embeddings by dim:", {r["dim"]: r["n"] for r in got})
    conn.close()
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
