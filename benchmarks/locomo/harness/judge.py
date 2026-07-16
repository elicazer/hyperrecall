"""Shared Gemini 2.5 Pro answerer and LoCoMo judge."""

from __future__ import annotations

import json
from typing import Any

from harness.metrics import bleu1, gold_string, token_f1
from harness.providers import ModelClient

MODEL = "gemini:gemini-2.5-pro"

ANSWER_PROMPT = """Answer the question about a long-term conversation using ONLY the memory context below. If the context is insufficient, reply exactly: I don't know. Keep the answer concise.

Memory context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """Grade a predicted answer to a question about a long-term conversation.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Return correct when the meaning matches (paraphrases are acceptable), partial when it has the right entity or concept but is incomplete, and wrong when contradictory, unrelated, or unanswered. A refusal is correct when the gold expects an unanswerable response.
Return only JSON: {{"label":"correct|partial|wrong","reason":"one short sentence"}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}


class GeminiJudge:
    """Own both model roles so every backend receives identical treatment."""

    def __init__(self) -> None:
        self.answerer = ModelClient(MODEL)
        self.judge = ModelClient(MODEL)

    def answer(self, context: str, question: str) -> str:
        return self.answerer.generate(ANSWER_PROMPT.format(context=context, question=question))

    def score(self, question: str, gold: Any, prediction: str) -> dict[str, Any]:
        gold_text = gold_string(gold)
        raw = self.judge.generate(
            JUDGE_PROMPT.format(question=question, gold=gold_text, prediction=prediction),
            json_schema=SCHEMA,
        )
        result = json.loads(raw)
        if result.get("label") not in {"correct", "partial", "wrong"}:
            raise ValueError(f"invalid judge label: {result.get('label')!r}")
        return {
            "label": result["label"],
            "reason": str(result.get("reason", "")),
            "f1": token_f1(prediction, gold_text),
            "bleu1": bleu1(prediction, gold_text),
        }

    def cost_record(self) -> dict[str, Any]:
        return {"answerer": self.answerer.cost(), "judge": self.judge.cost()}
