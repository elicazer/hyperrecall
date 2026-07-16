# Multi-system LoCoMo harness

This harness compares MeshMind, a raw-turn vector-RAG baseline, Mem0 Cloud,
and Zep Cloud with the same normalized LoCoMo conversation, retrieval-to-answer
prompt, Gemini 2.5 Pro answerer, and Gemini 2.5 Pro judge.

## Run conv-26

From the repository root, using the project virtual environment:

```bash
GEMINI_API_KEY=... .venv/bin/python benchmarks/locomo/harness/run_all.py \
  --conv 26 \
  --systems meshmind,vector_rag,mem0,zep \
  --run-id harness_pilot
```

The default run ID is a UTC timestamp. Each run writes one resumable JSONL file
per successful system and a `summary.json` containing overall and per-category
strict, lax, token-F1, and BLEU-1 scores. Reusing a run ID resumes completed
question indices.

The conv-26 MeshMind adapter reads the existing phase-1 mesh from
`runs/phase1/conv-26.embed.sqlite` (falling back to `conv-26.sqlite`). Vector RAG
embeds all raw turns locally with `sentence-transformers/all-MiniLM-L6-v2`.

## Paid systems and keys

Mem0 is the real hosted `MemoryClient` integration and requires both
`MEM0_API_KEY` and the `mem0ai` package. Zep remains an explicit Cloud stub
pending validation against the installed account/SDK combination. Without the
relevant keys, both adapters raise immediately and `run_all.py` records them in
`summary.json` under `skipped`; it never substitutes another memory system.

The earlier local Mem0 path was not used: it requires OpenAI for extraction and
embeddings and therefore is neither keyless nor free/local. No OpenAI, AWS, or
Bedrock calls are made by this harness.

## Methodology caveat

MeshMind phase-1 extraction cost is excluded because the adapter consumes the
pre-built benchmark mesh. The output calls this out in its cost record. Vector
RAG uses local embeddings. Gemini token usage is metered from API responses.
Hosted memory-system charges, when enabled, remain provider-metered.
