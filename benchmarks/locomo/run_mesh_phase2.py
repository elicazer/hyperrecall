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
        "--planner", choices=("legacy", "v2", "v2-rerank"), default="legacy",
        help="query-time retrieval planner (default: legacy merged retrieval)",
    )
    parser.add_argument(
        "--k-candidate", type=int, default=25,
        help="rerank: candidate edges retrieved before reranking (v2-rerank only)",
    )
    parser.add_argument(
        "--k-final", type=int, default=8,
        help="rerank: edges kept after reranking (v2-rerank only)",
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

    # v2-rerank makes two Gemini calls per question (batched rerank + answer),
    # which spikes the per-minute rate limit. Wrap the shared answerer with a
    # longer 429-aware backoff so a quota-limited run still completes.
    import time as _time
    _base_answer = P.answer

    def _resilient_answer(client, ctx, question, prompt_template=P.ANSWER_PROMPT):
        for attempt in range(6):
            out = _base_answer(client, ctx, question, prompt_template)
            if "RESOURCE_EXHAUSTED" not in out and "429" not in out:
                return out
            _time.sleep(min(60, 8 * (2 ** attempt)))
        return out

    if args.planner == "v2-rerank":
        P.answer = _resilient_answer

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

    rerank_pace = float(os.environ.get("RERANK_SLEEP", "5.0"))

    # Rerank scoring goes through the SAME new-SDK client as the answerer (the
    # deprecated google.generativeai path the planner ships trips the free-tier
    # RPM limit noticeably harder). One batched JSON call per question, with a
    # 429-aware wait since the RPM window recovers in ~60s.
    from google.genai import types as _gtypes  # noqa: E402

    def flash_score(prompt: str) -> str:
        for attempt in range(6):
            try:
                resp = client.models.generate_content(
                    model=P.MODEL,
                    contents=[_gtypes.Content(
                        role="user", parts=[_gtypes.Part.from_text(text=prompt)])],
                    config=_gtypes.GenerateContentConfig(
                        temperature=0.0,
                        thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
                        max_output_tokens=400,
                        response_mime_type="application/json",
                    ),
                )
                return resp.text or "{}"
            except Exception as exc:
                if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                    raise
                if attempt == 5:
                    raise
                _time.sleep(min(60, 15 * (attempt + 1)))
        return "{}"

    # Heuristic classification (llm=None) keeps this path at 2 Gemini calls per
    # question (rerank + answer); the reranker is what we are measuring here.
    from meshmind.query.planner import QueryPlanner  # noqa: E402
    rerank_planner = QueryPlanner(
        mesh.store, embed=mm_embed, curve=mesh.curve,
        llm=None, rerank_llm=flash_score, use_gemini=False,
    )

    def retrieve_v2_rerank(question: str) -> tuple[str, dict[str, object]]:
        if rerank_pace:
            _time.sleep(rerank_pace)  # throttle: 2 Gemini calls/question vs. RPM
        result = rerank_planner.recall(
            question,
            budget_tokens=None,
            k_hops=2,
            max_seeds=P.MESH_MAX_SEEDS,
            sim_rerank=P.MESH_SIM_RERANK,
            reinforce_on_access=False,
            rerank=True,
            k_candidate=args.k_candidate,
            k_final=args.k_final,
        )
        plan = result.plan
        rr = result.rerank or {}
        return result.to_context_string() or "(no relevant memory found)", {
            "path": "planner-v2-rerank",
            "question_class": plan.question_class,
            "question_kind": plan.question_kind,
            "entities": list(plan.entities),
            "sub_questions": list(plan.sub_questions),
            "n_nodes": len(result.nodes),
            "n_edges": len(result.results),
            "rerank_applied": rr.get("applied", False),
            "rerank_reason": rr.get("reason"),
            "rerank_n_candidates": rr.get("n_candidates", 0),
            "rerank_promoted_into_top_k": rr.get("promoted_into_top_k", 0),
            "rerank_deltas": rr.get("deltas", []),
        }

    limit = int(os.environ.get("PHASE2_LIMIT", "0")) or None
    if args.planner == "v2":
        P.OUT_DIR = P.ROOT / "runs" / "planner_v2"
        P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    elif args.planner == "v2-rerank":
        P.OUT_DIR = P.ROOT / "runs" / "rerank_v1"
        P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieve = {
        "v2": retrieve_v2,
        "v2-rerank": retrieve_v2_rerank,
    }.get(args.planner, dispatcher.retrieve)
    out = P.run_system(
        "meshmind", conv, client, retrieve,
        limit=limit, prompt_fn=dispatcher.prompt_for,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
