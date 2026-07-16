"""Re-run phase-2 answer generation for MeshMind ONLY.

Leaves the vector-RAG baseline predictions (conv-26.vector_rag.jsonl) untouched
so it stays a byte-identical fixed reference. Reuses phase2_retrieval's mesh
retrieval + Gemini answerer verbatim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import phase2_retrieval as P  # noqa: E402
from harness.load import load  # noqa: E402
from meshmind import Mesh  # noqa: E402


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    conv = next(c for c in load() if c.sample_id == P.CONV_ID)
    print(f"conv={P.CONV_ID} qas={len(conv.qa)}  mesh={P.MESH_DB.name}")
    print(f"routing: temporal_path={P.MESH_TEMPORAL_PATH} "
          f"temporal_prompt={P.MESH_TEMPORAL_PROMPT} sim_rerank={P.MESH_SIM_RERANK} "
          f"no_dates={P.MESH_NO_DATES} seeds={P.MESH_MAX_SEEDS}")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    embedder = SentenceTransformer(P.EMBEDDER_NAME)

    def mm_embed(text: str) -> np.ndarray:
        return embedder.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

    mesh = Mesh(str(P.MESH_DB), embed=mm_embed)
    dispatcher = P.MeshDispatcher(mesh, embedder, conv)

    limit = int(os.environ.get("PHASE2_LIMIT", "0")) or None
    out = P.run_system(
        "meshmind", conv, client, dispatcher.retrieve,
        limit=limit, prompt_fn=dispatcher.prompt_for,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
