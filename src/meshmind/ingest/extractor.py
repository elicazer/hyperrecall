"""Turn raw text into nodes + hyperedges.

MeshMind ingest decomposes a natural-language turn into atomic memory units
(nodes) and the typed, N-ary relations between them (hyperedges). Two backends
implement the same small interface:

* :class:`LLMExtractor` — the real pipeline. It asks Bedrock Claude Opus to
  decompose an utterance, and forces reliable structured JSON via a
  ``record_memory`` tool the model *must* call (Bedrock ``converse`` +
  ``toolChoice``). It validates the returned JSON against dataclass schemas,
  retries transient Bedrock errors with exponential backoff, and supports a
  ``mock_mode`` so tests and offline demos never touch the network.
* :class:`HeuristicExtractor` — the honest, deterministic fallback used when
  Bedrock is not configured. It is the original v0.0.1 stub, kept for
  offline/no-key environments only.

Both return an :class:`ExtractedMemory`, which :meth:`ExtractedMemory.to_extraction`
turns into the concrete ``nodes + hyperedges`` fragment that :class:`~meshmind.Mesh`
persists.

The legacy module-level :func:`extract` / :class:`Extraction` API is preserved
verbatim so existing callers (``Mesh.remember``) keep working unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..models import (
    EXPERIENCE,
    Hyperedge,
    HyperedgeMember,
    Node,
)

# --------------------------------------------------------------------------- #
# Legacy heuristic API (unchanged — Mesh.remember() depends on this surface).
# --------------------------------------------------------------------------- #


@dataclass
class Extraction:
    """The nodes and edges pulled out of one ingest call."""

    nodes: list[Node] = field(default_factory=list)
    hyperedges: list[Hyperedge] = field(default_factory=list)
    primary: Node | None = None


def extract(
    text: str,
    *,
    participants: list[str] | None = None,
    context: dict[str, Any] | None = None,
    confidence: float = 1.0,
    edge_type: str = EXPERIENCE,
    provenance: dict[str, Any] | None = None,
) -> Extraction:
    """Decompose one utterance into a small hypergraph fragment (heuristic)."""
    participants = participants or []
    context = context or {}
    ex = Extraction()

    primary = Node(text=text.strip(), kind="fact", confidence=confidence, metadata=dict(context))
    ex.nodes.append(primary)
    ex.primary = primary

    members: list[HyperedgeMember] = [HyperedgeMember(primary.id, role="statement", weight=1.0)]

    for name in participants:
        person = Node(text=name, kind="entity", confidence=confidence, metadata={"role": "participant"})
        ex.nodes.append(person)
        members.append(HyperedgeMember(person.id, role="participant", weight=0.8))

    # Promote salient context keys to their own nodes so they can be recalled.
    for key in ("topic", "project", "decision", "outcome"):
        val = context.get(key)
        if isinstance(val, str) and val.strip():
            cnode = Node(text=val.strip(), kind=key, confidence=confidence, metadata={"from": key})
            ex.nodes.append(cnode)
            members.append(HyperedgeMember(cnode.id, role=key, weight=0.7))

    # Only build the edge if it's genuinely a relation (arity >= 2).
    if len(members) >= 2:
        edge = Hyperedge(
            type=edge_type,
            members=members,
            confidence=confidence,
            provenance=provenance or {},
            metadata={"session": context.get("session")} if context.get("session") else {},
        )
        ex.hyperedges.append(edge)

    return ex


# --------------------------------------------------------------------------- #
# Structured-output schema (validated dataclasses — no pydantic dependency).
# --------------------------------------------------------------------------- #

ENTITY_KINDS = ("person", "project", "concept", "event", "place", "other")

#: Free-form hyperedge types the extractor is encouraged to emit. These are
#: conventions, not a closed enum (see ``models.HYPEREDGE_TYPES``).
HYPEREDGE_TYPE_HINTS = (
    "Discussion",
    "Decision",
    "Observation",
    "StateChange",
    "Event",
    EXPERIENCE,
)


class ExtractionError(ValueError):
    """Raised when the LLM returns JSON that does not match the schema."""


class TransientBedrockError(RuntimeError):
    """A retryable Bedrock failure (throttling, timeout, 5xx)."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ExtractionError(msg)


def _as_float(value: Any, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce to a float clamped to ``[lo, hi]``; fall back to ``default``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def parse_iso8601(value: Any) -> str | None:
    """Validate an ISO-8601 timestamp, returning a normalized string or ``None``.

    ``None``/empty is allowed (the model could not infer a time). A non-empty
    value that is not parseable is a schema violation.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    candidate = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        # Accept date-only ISO strings too (e.g. "2026-07-13").
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError as exc:
            raise ExtractionError(f"timestamp is not ISO-8601: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass
class ExtractedEntity:
    name: str
    kind: str = "other"
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, d: Any) -> "ExtractedEntity":
        _require(isinstance(d, dict), f"entity must be an object, got {type(d).__name__}")
        name = d.get("name")
        _require(isinstance(name, str) and name.strip(), "entity.name must be a non-empty string")
        kind = d.get("kind", "other")
        if kind not in ENTITY_KINDS:
            kind = "other"
        return cls(name=name.strip(), kind=kind, confidence=_as_float(d.get("confidence"), 1.0))


@dataclass
class ExtractedParticipant:
    entity_name: str
    role: str = "participant"
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: Any) -> "ExtractedParticipant":
        _require(isinstance(d, dict), f"participant must be an object, got {type(d).__name__}")
        name = d.get("entity_name") or d.get("name")
        _require(isinstance(name, str) and name.strip(), "participant.entity_name required")
        role = d.get("role") or "participant"
        _require(isinstance(role, str), "participant.role must be a string")
        return cls(
            entity_name=name.strip(),
            role=role.strip() or "participant",
            weight=_as_float(d.get("weight"), 1.0, hi=2.0),
        )


@dataclass
class ExtractedHyperedge:
    type: str
    participants: list[ExtractedParticipant] = field(default_factory=list)
    timestamp: str | None = None
    confidence: float = 1.0
    source_text: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "ExtractedHyperedge":
        _require(isinstance(d, dict), "hyperedge must be an object")
        etype = d.get("type")
        _require(isinstance(etype, str) and etype.strip(), "hyperedge.type must be a non-empty string")
        parts = d.get("participants") or []
        _require(isinstance(parts, list), "hyperedge.participants must be a list")
        provenance = d.get("provenance") or {}
        source_text = ""
        if isinstance(provenance, dict):
            source_text = str(provenance.get("source_text") or "")
        return cls(
            type=etype.strip(),
            participants=[ExtractedParticipant.from_dict(p) for p in parts],
            timestamp=parse_iso8601(d.get("timestamp")),
            confidence=_as_float(d.get("confidence"), 1.0),
            source_text=source_text,
        )


@dataclass
class RelationHint:
    """A soft pointer to an existing memory that this turn conflicts with /
    replaces. Resolving the hint to a concrete node is out of scope here — the
    hint is preserved as edge/node metadata for later reconciliation."""

    target_hyperedge_or_node_hint: str
    reason: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "RelationHint":
        _require(isinstance(d, dict), "relation hint must be an object")
        target = d.get("target_hyperedge_or_node_hint") or d.get("target") or ""
        _require(isinstance(target, str) and target.strip(), "relation hint needs a target")
        reason = d.get("reason") or ""
        return cls(target_hyperedge_or_node_hint=target.strip(), reason=str(reason))

    def to_dict(self) -> dict[str, str]:
        return {
            "target_hyperedge_or_node_hint": self.target_hyperedge_or_node_hint,
            "reason": self.reason,
        }


@dataclass
class ExtractedMemory:
    """The validated, backend-agnostic result of extracting one turn.

    :meth:`to_extraction` lowers it into the concrete ``nodes + hyperedges``
    fragment that :class:`~meshmind.Mesh` persists.
    """

    source_text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    hyperedge: ExtractedHyperedge | None = None
    contradictions: list[RelationHint] = field(default_factory=list)
    supersedes: list[RelationHint] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    # -- validation ---------------------------------------------------------
    @classmethod
    def from_tool_input(cls, payload: Any, *, source_text: str) -> "ExtractedMemory":
        """Validate the ``record_memory`` tool payload into an ExtractedMemory."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ExtractionError(f"tool input was not valid JSON: {exc}") from exc
        _require(isinstance(payload, dict), "tool input must be a JSON object")

        entities_raw = payload.get("entities") or []
        _require(isinstance(entities_raw, list), "entities must be a list")
        entities = [ExtractedEntity.from_dict(e) for e in entities_raw]

        edge_raw = payload.get("hyperedge")
        hyperedge = None
        if edge_raw is not None:
            hyperedge = ExtractedHyperedge.from_dict(edge_raw)
            if not hyperedge.source_text:
                hyperedge.source_text = source_text

        contradictions = [RelationHint.from_dict(c) for c in (payload.get("contradictions") or [])]
        supersedes = [RelationHint.from_dict(s) for s in (payload.get("supersedes") or [])]

        return cls(
            source_text=source_text,
            entities=entities,
            hyperedge=hyperedge,
            contradictions=contradictions,
            supersedes=supersedes,
            raw=payload,
        )

    # -- lowering -----------------------------------------------------------
    def to_extraction(
        self,
        *,
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Extraction:
        """Turn the extracted memory into concrete nodes + a hyperedge."""
        context = context or {}
        edge_conf = confidence if confidence is not None else (
            self.hyperedge.confidence if self.hyperedge else 1.0
        )
        ex = Extraction()

        primary = Node(
            text=self.source_text.strip(),
            kind="fact",
            confidence=edge_conf,
            metadata={
                **context,
                **({"timestamp": self.hyperedge.timestamp} if self.hyperedge and self.hyperedge.timestamp else {}),
            },
        )
        ex.nodes.append(primary)
        ex.primary = primary

        members: list[HyperedgeMember] = [HyperedgeMember(primary.id, role="statement", weight=1.0)]

        # role/weight lookup from the hyperedge's participant list
        role_by_entity: dict[str, ExtractedParticipant] = {}
        if self.hyperedge:
            for p in self.hyperedge.participants:
                role_by_entity[p.entity_name.lower()] = p

        for ent in self.entities:
            enode = Node(
                text=ent.name,
                kind=ent.kind,
                confidence=ent.confidence,
                metadata={"entity_kind": ent.kind},
            )
            ex.nodes.append(enode)
            part = role_by_entity.get(ent.name.lower())
            role = part.role if part else "participant"
            weight = part.weight if part else 0.8
            members.append(HyperedgeMember(enode.id, role=role, weight=weight))

        # A participant named in the hyperedge but not in `entities` still counts.
        seen = {e.name.lower() for e in self.entities}
        if self.hyperedge:
            for p in self.hyperedge.participants:
                if p.entity_name.lower() in seen:
                    continue
                enode = Node(text=p.entity_name, kind="entity", confidence=edge_conf)
                ex.nodes.append(enode)
                members.append(HyperedgeMember(enode.id, role=p.role, weight=p.weight))

        if len(members) >= 2:
            edge_type = self.hyperedge.type if self.hyperedge else EXPERIENCE
            meta: dict[str, Any] = {}
            if self.hyperedge and self.hyperedge.timestamp:
                meta["timestamp"] = self.hyperedge.timestamp
            if self.contradictions:
                meta["contradictions"] = [c.to_dict() for c in self.contradictions]
            if self.supersedes:
                meta["supersedes"] = [s.to_dict() for s in self.supersedes]
            prov = dict(provenance or {})
            prov.setdefault("source_text", self.source_text)
            prov.setdefault("extractor", "llm")
            edge = Hyperedge(
                type=edge_type,
                members=members,
                confidence=edge_conf,
                provenance=prov,
                metadata=meta,
            )
            ex.hyperedges.append(edge)

        return ex


# --------------------------------------------------------------------------- #
# Shared deterministic parser (fallback + offline mock).
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(
    r"\b(?:on\s+)?"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:,?\s+(\d{4}))?",
    re.IGNORECASE,
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
# Capitalized tokens that are almost certainly not entity names.
_STOP = {"The", "A", "An", "On", "In", "At", "It", "They", "We", "I", "He", "She"}


def _guess_timestamp(text: str) -> str | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    month = _MONTHS[m.group(1).lower()]
    day = int(m.group(2))
    year = int(m.group(3)) if m.group(3) else datetime.now(timezone.utc).year
    try:
        return datetime(year, month, day, tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _guess_type(text: str) -> str:
    low = text.lower()
    if "decid" in low or "decision" in low or "chose" in low:
        return "Decision"
    if "discuss" in low or "talked" in low or "asked" in low or "met" in low:
        return "Discussion"
    if "became" in low or "changed" in low or "now" in low or "moved" in low:
        return "StateChange"
    if "happened" in low or "event" in low:
        return "Event"
    return "Observation"


def _guess_entities(text: str, participants: list[str]) -> list[str]:
    names = list(dict.fromkeys(participants))  # honor explicit participants, de-duped
    for token in re.findall(r"\b[A-Z][a-zA-Z]+\b", text):
        if token in _STOP or token in _MONTHS or token.capitalize() in _MONTHS:
            continue
        if token not in names:
            names.append(token)
    return names


def heuristic_memory(
    text: str,
    *,
    participants: list[str] | None = None,
    context: dict[str, Any] | None = None,
    confidence: float = 1.0,
) -> ExtractedMemory:
    """Deterministic, network-free extraction. Used by the fallback extractor
    and as the default offline mock for :class:`LLMExtractor`."""
    text = text.strip()
    context = context or {}
    names = _guess_entities(text, participants or [])
    entities = [ExtractedEntity(name=n, kind="other", confidence=0.6) for n in names]
    edge = ExtractedHyperedge(
        type=_guess_type(text) if text else "Observation",
        participants=[ExtractedParticipant(entity_name=n, role="participant", weight=0.8) for n in names],
        timestamp=_guess_timestamp(text),
        confidence=confidence,
        source_text=text,
    )
    return ExtractedMemory(
        source_text=text,
        entities=entities,
        hyperedge=edge if names else None,
        raw={"backend": "heuristic"},
    )


# --------------------------------------------------------------------------- #
# Extractor backends.
# --------------------------------------------------------------------------- #


class HeuristicExtractor:
    """Deterministic fallback extractor (the original v0.0.1 stub, class-wrapped).

    Fallback-only: used when Bedrock is not configured. It does *no* semantic
    typing beyond simple keyword rules. Prefer :class:`LLMExtractor` whenever a
    Bedrock key is available.
    """

    backend = "heuristic"

    def extract(
        self,
        text: str,
        *,
        participants: list[str] | None = None,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
    ) -> ExtractedMemory:
        return heuristic_memory(
            text, participants=participants, context=context, confidence=confidence
        )


# The tool schema we force the model to call. Kept module-level so tests and
# the demo can introspect it.
RECORD_MEMORY_TOOL: dict[str, Any] = {
    "name": "record_memory",
    "description": (
        "Record the atomic memory extracted from a single natural-language turn "
        "as a hypergraph fragment: the entities involved, one typed N-ary "
        "hyperedge binding them with roles, plus any contradiction/supersession "
        "hints against prior memory. You MUST call this tool exactly once."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "enum": list(ENTITY_KINDS)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["name", "kind"],
                    },
                },
                "hyperedge": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "participants": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity_name": {"type": "string"},
                                    "role": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                                "required": ["entity_name", "role"],
                            },
                        },
                        "timestamp": {
                            "type": ["string", "null"],
                            "description": "ISO-8601 if inferable, else null",
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "provenance": {
                            "type": "object",
                            "properties": {"source_text": {"type": "string"}},
                        },
                    },
                    "required": ["type", "participants"],
                },
                "contradictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_hyperedge_or_node_hint": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["target_hyperedge_or_node_hint", "reason"],
                    },
                },
                "supersedes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_hyperedge_or_node_hint": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["target_hyperedge_or_node_hint", "reason"],
                    },
                },
            },
            "required": ["entities", "hyperedge"],
        }
    },
}

_SYSTEM_PROMPT = (
    "You are MeshMind's ingestion engine. Decompose one natural-language turn "
    "into a hypergraph memory: the distinct entities (people, projects, "
    "concepts, events, places), one typed hyperedge that binds them with roles "
    "(who did what), a timestamp if one is stated or inferable, and any "
    "contradiction/supersession hints against prior knowledge. Keep entities "
    "atomic. Prefer hyperedge types like Discussion, Decision, Observation, "
    "StateChange, Event. Always call the record_memory tool exactly once."
)

# Bedrock error codes we treat as transient/retryable.
_TRANSIENT_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ServiceQuotaExceededException",
}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientBedrockError):
        return True
    if exc.__class__.__name__ in _TRANSIENT_CODES:
        return True
    # botocore ClientError carries the AWS error code in .response
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_CODES:
            return True
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(status, int) and status >= 500:
            return True
    return False


class LLMExtractor:
    """Real LLM-based extractor backed by AWS Bedrock (Claude Opus).

    Parameters
    ----------
    model:
        Bedrock model id. Defaults to ``global.anthropic.claude-opus-4-8``.
    region:
        AWS region for the ``bedrock-runtime`` client (default ``us-east-1``).
    mock_mode:
        When ``True`` no network call is made. If ``mock_response`` is provided
        it is validated as the tool payload; otherwise a deterministic
        heuristic parse is returned. Used for CI and offline demos.
    mock_response:
        A canned ``record_memory`` tool-input dict returned in ``mock_mode``.
    max_retries:
        Number of retries on transient Bedrock errors (exponential backoff).
    client:
        Inject a pre-built ``bedrock-runtime`` client (mostly for tests).
    """

    backend = "llm"

    def __init__(
        self,
        *,
        model: str = "global.anthropic.claude-opus-4-8",
        region: str = "us-east-1",
        mock_mode: bool = False,
        mock_response: dict[str, Any] | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        self.model = model
        self.region = region
        self.mock_mode = mock_mode
        self.mock_response = mock_response
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_tokens = max_tokens
        self._client = client

    # -- client -------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # lazy: not needed for mock_mode / fallback

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    # -- public API ---------------------------------------------------------
    def extract(
        self,
        text: str,
        *,
        participants: list[str] | None = None,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
    ) -> ExtractedMemory:
        source = text.strip()
        if not source:
            # Nothing to extract; return an empty memory (no hyperedge).
            return ExtractedMemory(source_text="", raw={"backend": self.backend, "empty": True})

        if self.mock_mode:
            if self.mock_response is not None:
                return ExtractedMemory.from_tool_input(self.mock_response, source_text=source)
            return heuristic_memory(
                source, participants=participants, context=context, confidence=confidence
            )

        payload = self._invoke_with_retries(source, participants=participants, context=context)
        return ExtractedMemory.from_tool_input(payload, source_text=source)

    # -- bedrock ------------------------------------------------------------
    def _build_request(
        self,
        text: str,
        *,
        participants: list[str] | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        hint = ""
        if participants:
            hint += f"\nKnown participants: {', '.join(participants)}."
        if context:
            hint += f"\nContext: {json.dumps(context, default=str)}."
        user_text = f"Turn to ingest:\n{text}{hint}"
        return {
            "modelId": self.model,
            "system": [{"text": _SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "toolConfig": {
                "tools": [{"toolSpec": RECORD_MEMORY_TOOL}],
                "toolChoice": {"tool": {"name": "record_memory"}},
            },
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }

    def _invoke_with_retries(
        self,
        text: str,
        *,
        participants: list[str] | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request = self._build_request(text, participants=participants, context=context)
        client = self._get_client()
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.converse(**request)
                return self._extract_tool_input(response)
            except ExtractionError:
                raise  # a schema problem won't fix itself on retry
            except BaseException as exc:  # noqa: BLE001 - we re-raise below
                last_exc = exc
                if not _is_transient(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(self.backoff_base * (2 ** attempt))
        # Unreachable, but keeps type-checkers happy.
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _extract_tool_input(response: Any) -> dict[str, Any]:
        """Pull the ``record_memory`` tool input out of a converse() response."""
        _require(isinstance(response, dict), "bedrock response was not a dict")
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        _require(isinstance(content, list), "response content must be a list")
        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                tool_use = block["toolUse"]
                if tool_use.get("name") == "record_memory":
                    return tool_use.get("input", {})
        raise ExtractionError("model did not call the record_memory tool")


# --------------------------------------------------------------------------- #
# Backend selection.
# --------------------------------------------------------------------------- #


def bedrock_available() -> bool:
    """True if a Bedrock key is present and boto3 is importable."""
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return False
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False
    return True


def choose_extractor(
    *,
    use_llm: bool | None = None,
    mock_mode: bool = False,
    **kwargs: Any,
) -> HeuristicExtractor | LLMExtractor:
    """Pick an extractor. ``mock_mode`` forces a network-free LLMExtractor;
    otherwise use the LLM when Bedrock is configured, else the heuristic."""
    if mock_mode:
        return LLMExtractor(mock_mode=True, **kwargs)
    if use_llm is None:
        use_llm = bedrock_available()
    if use_llm:
        return LLMExtractor(mock_mode=False, **kwargs)
    return HeuristicExtractor()
