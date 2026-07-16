"""Contamination probe: ask Gemini 25 QAs with NO context.
If it answers correctly at high rate, the model has memorized LoCoMo."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())
from google import genai
from google.genai import types
from harness.load import load

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
conv = next(c for c in load() if c.sample_id == "conv-26")

# 25 well-spaced QAs
step = max(1, len(conv.qa) // 25)
sample = conv.qa[::step][:25]
print(f"probing {len(sample)} QAs cold (no context)")

hits = 0
for qa in sample:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(
            text=f"Answer briefly. If unsure, say 'I don't know.'\n\nQuestion: {qa.question}\nAnswer:")])],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=80),
    )
    pred = (resp.text or "").strip().lower()
    gold = str(qa.answer).lower()
    # crude token overlap check
    gold_toks = set(gold.split())
    pred_toks = set(pred.split())
    overlap = len(gold_toks & pred_toks) / max(1, len(gold_toks))
    hit = overlap >= 0.5 or gold in pred or pred in gold
    if hit and "don't know" not in pred and "i do not know" not in pred:
        hits += 1
    print(f"  cat={qa.category}  Q: {qa.question[:60]}")
    print(f"    gold: {qa.answer}")
    print(f"    pred: {resp.text.strip()[:80]}")
    print(f"    hit={hit}")
print(f"\ncold-answer hit rate: {hits}/{len(sample)} = {hits/len(sample)*100:.1f}%")
