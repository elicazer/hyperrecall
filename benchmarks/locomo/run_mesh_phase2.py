"""Re-run phase-2 answer generation for MeshMind ONLY.

Leaves the vector-RAG baseline predictions (conv-26.vector_rag.jsonl) untouched
so it stays a byte-identical fixed reference. Reuses phase2_retrieval's mesh
retrieval + Gemini answerer verbatim.
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner", choices=("legacy", "v2", "v2-chain"), default="legacy",
        help="query-time retrieval planner (default: legacy merged retrieval)",
    )
    args = parser.parse_args(argv)
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    conv = next(c for c in load() if c.sample_id == P.CONV_ID)
    print(f"conv={P.CONV_ID} qas={len(conv.qa)}  mesh={P.MESH_DB.name} planner={args.planner}")
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

    def retrieve_v2(question: str) -> tuple[str, dict[str, object]]:
        result = mesh.recall(
            question,
            plan="v2",
            budget_tokens=None,
            k_hops=2,
            max_seeds=P.MESH_MAX_SEEDS,
            sim_rerank=P.MESH_SIM_RERANK,
            reinforce_on_access=False,
        )
        plan = result.plan
        return result.to_context_string() or "(no relevant memory found)", {
            "path": "planner-v2",
            "question_class": plan.question_class,
            "question_kind": plan.question_kind,
            "entities": list(plan.entities),
            "sub_questions": list(plan.sub_questions),
            "n_nodes": len(result.nodes),
            "n_edges": len(result.results),
        }

    def retrieve_v2_chain(question: str) -> tuple[str, dict[str, object]]:
        result = mesh.recall(
            question,
            plan="v2-chain",
            budget_tokens=None,
            k_hops=2,
            max_seeds=P.MESH_MAX_SEEDS,
            sim_rerank=P.MESH_SIM_RERANK,
            reinforce_on_access=False,
        )
        plan = result.plan
        return result.to_context_string() or "(no relevant memory found)", {
            "path": "planner-v2-chain",
            "question_class": plan.question_class,
            "question_kind": plan.question_kind,
            "entities": list(plan.entities),
            "sub_questions": list(plan.sub_questions),
            "n_nodes": len(result.nodes),
            "n_edges": len(result.results),
            "explanation": result.explanation,
        }

    limit = int(os.environ.get("PHASE2_LIMIT", "0")) or None
    if args.planner == "v2":
        P.OUT_DIR = P.ROOT / "runs" / "planner_v2"
        P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    elif args.planner == "v2-chain":
        P.OUT_DIR = P.ROOT / "runs" / "chain_v1"
        P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieve = {
        "legacy": dispatcher.retrieve,
        "v2": retrieve_v2,
        "v2-chain": retrieve_v2_chain,
    }[args.planner]
    out = P.run_system(
        "meshmind", conv, client, retrieve,
        limit=limit, prompt_fn=dispatcher.prompt_for,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
