"""Deterministic LoCoMo lexical metrics."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"\b\w+\b")


def tokens(value: str) -> list[str]:
    return _WORD_RE.findall((value or "").lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    precision = sum((Counter(p) & Counter(g)).values()) / len(p)
    brevity = 1.0 if len(p) >= len(g) else math.exp(1 - len(g) / len(p))
    return brevity * precision


def gold_string(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " / ".join(map(str, value))
    return str(value)
