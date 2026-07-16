"""Load and normalize the LoCoMo dataset.

The raw file is snap-research/locomo's data/locomo10.json (10 conversations).
This module reshapes it into a form our harness likes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_DATA = Path(__file__).resolve().parents[1] / "repo" / "data" / "locomo10.json"


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str  # e.g. "D1:3" — session 1, turn 3


@dataclass
class Session:
    index: int
    date_time: str
    turns: list[Turn]


@dataclass
class QA:
    question: str
    answer: Any  # sometimes string, sometimes list — see raw file
    category: int  # 1..5
    evidence: list[str] = field(default_factory=list)  # dia_ids that contain the answer


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa: list[QA]

    def to_context_text(self) -> str:
        """Full conversation flattened chronologically. Used for baselines that
        need the raw dialog (vector RAG, full-context)."""
        parts: list[str] = []
        for sess in self.sessions:
            parts.append(f"--- Session {sess.index} ({sess.date_time}) ---")
            for t in sess.turns:
                parts.append(f"[{t.dia_id}] {t.speaker}: {t.text}")
        return "\n".join(parts)


def load(path: Path | str = REPO_DATA) -> list[Conversation]:
    raw = json.loads(Path(path).read_text())
    out: list[Conversation] = []
    for sample in raw:
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        # Session keys look like session_1, session_2, ...; timestamps live
        # in session_1_date_time, etc. Sort numerically.
        session_indices = sorted(
            int(k.split("_")[1])
            for k in conv
            if k.startswith("session_")
            and not k.endswith("date_time")
            and isinstance(conv[k], list)
        )
        sessions: list[Session] = []
        for i in session_indices:
            turns_raw = conv[f"session_{i}"]
            dt = conv.get(f"session_{i}_date_time", "")
            turns = [
                Turn(
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    dia_id=t.get("dia_id", ""),
                )
                for t in turns_raw
            ]
            sessions.append(Session(index=i, date_time=dt, turns=turns))

        qa = [
            QA(
                question=q["question"],
                answer=q.get("answer", ""),
                category=int(q.get("category", 0)),
                evidence=list(q.get("evidence", []) or []),
            )
            for q in sample.get("qa", [])
        ]

        out.append(
            Conversation(
                sample_id=sample["sample_id"],
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                sessions=sessions,
                qa=qa,
            )
        )
    return out


def iter_turns(conv: Conversation) -> Iterable[tuple[Session, Turn]]:
    for sess in conv.sessions:
        for t in sess.turns:
            yield sess, t


if __name__ == "__main__":
    convs = load()
    print(f"loaded {len(convs)} conversations")
    for c in convs:
        n_turns = sum(len(s.turns) for s in c.sessions)
        print(
            f"  {c.sample_id}: {c.speaker_a}/{c.speaker_b}  "
            f"{len(c.sessions)} sessions  {n_turns} turns  {len(c.qa)} QAs"
        )
