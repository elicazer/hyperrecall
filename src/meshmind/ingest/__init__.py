"""Ingestion: raw text -> nodes + hyperedges.

Two backends behind one interface: :class:`LLMExtractor` (real, Bedrock Claude
Opus, structured output) and :class:`HeuristicExtractor` (deterministic
fallback). :func:`choose_extractor` picks based on whether Bedrock is
configured. The legacy :func:`extract` / :class:`Extraction` API is retained.
"""

from .extractor import (
    ExtractedEntity,
    ExtractedHyperedge,
    ExtractedMemory,
    ExtractedParticipant,
    Extraction,
    ExtractionError,
    HeuristicExtractor,
    LLMExtractor,
    RelationHint,
    bedrock_available,
    choose_extractor,
    extract,
)

__all__ = [
    "Extraction",
    "extract",
    "ExtractedMemory",
    "ExtractedEntity",
    "ExtractedHyperedge",
    "ExtractedParticipant",
    "RelationHint",
    "ExtractionError",
    "HeuristicExtractor",
    "LLMExtractor",
    "choose_extractor",
    "bedrock_available",
]
