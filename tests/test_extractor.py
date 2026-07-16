"""Tests for the real LLM-based ingest pipeline (Bedrock Claude Opus).

Every test runs offline: either ``mock_mode=True`` (deterministic/canned
responses) or against an injected fake Bedrock client. No network calls.
"""

from __future__ import annotations

import pytest

from meshmind import Mesh
from meshmind.ingest.extractor import (
    ExtractedMemory,
    ExtractionError,
    HeuristicExtractor,
    LLMExtractor,
    RECORD_MEMORY_TOOL,
    bedrock_available,
    choose_extractor,
    heuristic_memory,
    parse_iso8601,
)

# --------------------------------------------------------------------------- #
# Canned "record_memory" tool payloads (what the model would return).
# --------------------------------------------------------------------------- #

TEDX_PAYLOAD = {
    "entities": [
        {"name": "Eli", "kind": "person", "confidence": 0.98},
        {"name": "David", "kind": "person", "confidence": 0.97},
        {"name": "TEDx application", "kind": "project", "confidence": 0.9},
        {"name": "ShapeForge", "kind": "project", "confidence": 0.85},
    ],
    "hyperedge": {
        "type": "Decision",
        "participants": [
            {"entity_name": "Eli", "role": "decider", "weight": 1.0},
            {"entity_name": "David", "role": "decider", "weight": 1.0},
            {"entity_name": "TEDx application", "role": "topic", "weight": 0.7},
            {"entity_name": "ShapeForge", "role": "subject", "weight": 0.6},
        ],
        "timestamp": "2026-07-13",
        "confidence": 0.92,
        "provenance": {"source_text": "On July 13, Eli and David decided ..."},
    },
    "contradictions": [],
    "supersedes": [],
}


def fake_converse_response(tool_input: dict) -> dict:
    """Shape of a Bedrock converse() response carrying a tool_use block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"name": "record_memory", "toolUseId": "t1", "input": tool_input}}
                ],
            }
        },
        "stopReason": "tool_use",
    }


class FakeBedrockClient:
    """Minimal stand-in for a bedrock-runtime client's converse()."""

    def __init__(self, response: dict | None = None):
        self.response = response if response is not None else fake_converse_response(TEDX_PAYLOAD)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ThrottleException(Exception):
    """Emulates a Bedrock ThrottlingException by class name."""


class FlakyBedrockClient:
    """Raises a transient error `fail_times` times, then succeeds."""

    def __init__(self, fail_times: int, response: dict | None = None):
        self.fail_times = fail_times
        self.response = response if response is not None else fake_converse_response(TEDX_PAYLOAD)
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ThrottleException("slow down")
        return self.response


# Rename ThrottleException so _is_transient recognizes it by name.
ThrottleException.__name__ = "ThrottlingException"


# --------------------------------------------------------------------------- #
# 1. Entity extraction
# --------------------------------------------------------------------------- #


def test_entity_extraction_from_canned_payload():
    ext = LLMExtractor(mock_mode=True, mock_response=TEDX_PAYLOAD)
    mem = ext.extract("On July 13, Eli and David decided on the ShapeForge narrative.")
    names = {e.name for e in mem.entities}
    assert {"Eli", "David"} <= names
    kinds = {e.name: e.kind for e in mem.entities}
    assert kinds["Eli"] == "person"
    assert kinds["ShapeForge"] == "project"


def test_entity_kind_falls_back_to_other_for_unknown_kind():
    payload = {
        "entities": [{"name": "Zorp", "kind": "alien", "confidence": 0.5}],
        "hyperedge": {"type": "Observation", "participants": [], "timestamp": None},
    }
    mem = ExtractedMemory.from_tool_input(payload, source_text="Zorp appeared.")
    assert mem.entities[0].kind == "other"


# --------------------------------------------------------------------------- #
# 2. Hyperedge participant roles
# --------------------------------------------------------------------------- #


def test_hyperedge_participant_roles_and_weights():
    ext = LLMExtractor(mock_mode=True, mock_response=TEDX_PAYLOAD)
    mem = ext.extract("...")
    roles = {p.entity_name: p.role for p in mem.hyperedge.participants}
    assert roles["Eli"] == "decider"
    assert roles["TEDx application"] == "topic"
    weights = {p.entity_name: p.weight for p in mem.hyperedge.participants}
    assert weights["Eli"] == 1.0


def test_roles_survive_lowering_into_hyperedge():
    ext = LLMExtractor(mock_mode=True, mock_response=TEDX_PAYLOAD)
    mem = ext.extract("...")
    ex = mem.to_extraction()
    assert len(ex.hyperedges) == 1
    edge = ex.hyperedges[0]
    # arity = 1 statement + 4 entities
    assert edge.arity == 5
    roles = {m.role for m in edge.members}
    assert "statement" in roles and "decider" in roles and "topic" in roles


# --------------------------------------------------------------------------- #
# 3. Contradictions
# --------------------------------------------------------------------------- #


def test_contradictions_parsed_and_carried_to_edge_metadata():
    payload = {
        "entities": [{"name": "Eli", "kind": "person"}],
        "hyperedge": {
            "type": "StateChange",
            "participants": [{"entity_name": "Eli", "role": "subject"}],
            "timestamp": None,
        },
        "contradictions": [
            {"target_hyperedge_or_node_hint": "Eli lives in Boston", "reason": "now Newport"}
        ],
    }
    mem = ExtractedMemory.from_tool_input(payload, source_text="Eli moved to Newport.")
    assert len(mem.contradictions) == 1
    assert mem.contradictions[0].reason == "now Newport"
    edge = mem.to_extraction().hyperedges[0]
    assert edge.metadata["contradictions"][0]["target_hyperedge_or_node_hint"] == "Eli lives in Boston"


# --------------------------------------------------------------------------- #
# 4. Supersedes
# --------------------------------------------------------------------------- #


def test_supersedes_parsed_and_carried():
    payload = {
        "entities": [{"name": "MeshMind", "kind": "project"}],
        "hyperedge": {
            "type": "StateChange",
            "participants": [{"entity_name": "MeshMind", "role": "subject"}],
            "timestamp": None,
        },
        "supersedes": [
            {"target_hyperedge_or_node_hint": "MeshMind is v0.0.1", "reason": "now v0.1.0"}
        ],
    }
    mem = ExtractedMemory.from_tool_input(payload, source_text="MeshMind is now v0.1.0.")
    assert mem.supersedes[0].target_hyperedge_or_node_hint == "MeshMind is v0.0.1"
    edge = mem.to_extraction().hyperedges[0]
    assert edge.metadata["supersedes"][0]["reason"] == "now v0.1.0"


# --------------------------------------------------------------------------- #
# 5. Timestamp parsing
# --------------------------------------------------------------------------- #


def test_timestamp_date_only_normalized_to_iso():
    assert parse_iso8601("2026-07-13").startswith("2026-07-13T00:00:00")


def test_timestamp_full_iso_with_z():
    out = parse_iso8601("2026-07-13T20:00:00Z")
    assert out.startswith("2026-07-13T20:00:00")
    assert "+00:00" in out


def test_timestamp_null_is_allowed():
    assert parse_iso8601(None) is None
    assert parse_iso8601("") is None


def test_timestamp_malformed_raises():
    with pytest.raises(ExtractionError):
        parse_iso8601("not-a-date")


def test_timestamp_flows_from_payload_to_memory():
    ext = LLMExtractor(mock_mode=True, mock_response=TEDX_PAYLOAD)
    mem = ext.extract("...")
    assert mem.hyperedge.timestamp.startswith("2026-07-13")


# --------------------------------------------------------------------------- #
# 6. Malformed LLM output handling
# --------------------------------------------------------------------------- #


def test_malformed_missing_hyperedge_type_raises():
    payload = {"entities": [], "hyperedge": {"participants": []}}
    with pytest.raises(ExtractionError):
        ExtractedMemory.from_tool_input(payload, source_text="x")


def test_malformed_entities_not_a_list_raises():
    payload = {"entities": "Eli", "hyperedge": {"type": "Observation", "participants": []}}
    with pytest.raises(ExtractionError):
        ExtractedMemory.from_tool_input(payload, source_text="x")


def test_malformed_non_json_string_raises():
    with pytest.raises(ExtractionError):
        ExtractedMemory.from_tool_input("{not json", source_text="x")


def test_model_that_never_calls_tool_raises():
    ext = LLMExtractor(client=FakeBedrockClient(response={"output": {"message": {"content": [{"text": "hi"}]}}}))
    with pytest.raises(ExtractionError):
        ext.extract("Eli met David.")


# --------------------------------------------------------------------------- #
# 7. Retry logic
# --------------------------------------------------------------------------- #


def test_retry_succeeds_after_transient_errors():
    flaky = FlakyBedrockClient(fail_times=2)
    ext = LLMExtractor(client=flaky, max_retries=3, backoff_base=0.0)
    mem = ext.extract("On July 13, Eli and David decided.")
    assert flaky.calls == 3  # 2 failures + 1 success
    assert any(e.name == "Eli" for e in mem.entities)


def test_retry_gives_up_after_max_retries():
    flaky = FlakyBedrockClient(fail_times=99)
    ext = LLMExtractor(client=flaky, max_retries=2, backoff_base=0.0)
    with pytest.raises(Exception):
        ext.extract("Eli met David.")
    assert flaky.calls == 3  # initial + 2 retries


def test_schema_error_is_not_retried():
    bad = fake_converse_response({"entities": "oops", "hyperedge": {"type": "X", "participants": []}})
    client = FakeBedrockClient(response=bad)
    ext = LLMExtractor(client=client, max_retries=3, backoff_base=0.0)
    with pytest.raises(ExtractionError):
        ext.extract("Eli met David.")
    assert len(client.calls) == 1  # not retried


# --------------------------------------------------------------------------- #
# 8. Empty input
# --------------------------------------------------------------------------- #


def test_empty_input_returns_empty_memory_no_network():
    # No client, not mock_mode: empty text must short-circuit before any call.
    ext = LLMExtractor()
    mem = ext.extract("   ")
    assert mem.source_text == ""
    assert mem.hyperedge is None
    assert mem.entities == []


def test_empty_memory_lowers_to_single_node_no_edge():
    ext = LLMExtractor()
    ex = ext.extract("").to_extraction()
    assert len(ex.nodes) == 1
    assert ex.hyperedges == []


# --------------------------------------------------------------------------- #
# 9. Multi-entity input (deterministic default mock)
# --------------------------------------------------------------------------- #


def test_default_mock_multi_entity_extraction():
    ext = LLMExtractor(mock_mode=True)  # no canned response -> heuristic parse
    mem = ext.extract("On July 13, Eli and David discussed the TEDx application.")
    names = {e.name for e in mem.entities}
    assert "Eli" in names and "David" in names
    assert mem.hyperedge is not None
    assert mem.hyperedge.type == "Discussion"
    assert mem.hyperedge.timestamp.startswith("2026-07-13")


def test_default_mock_detects_decision_type():
    mem = heuristic_memory("Eli and David decided to focus on ShapeForge.")
    assert mem.hyperedge.type == "Decision"


# --------------------------------------------------------------------------- #
# 10. Integration with Mesh.ingest_text
# --------------------------------------------------------------------------- #


def test_mesh_ingest_text_persists_and_recalls():
    mesh = Mesh(":memory:")
    mem = mesh.ingest_text(
        "On July 13, Eli and David decided on the ShapeForge narrative.",
        mock_mode=True,
    )
    assert isinstance(mem, ExtractedMemory)
    stats = mesh.stats()
    assert stats["nodes"] >= 3  # statement + entities
    assert stats["hyperedges"] >= 1
    result = mesh.recall("ShapeForge")
    assert result.nodes
    mesh.close()


def test_mesh_ingest_text_with_injected_extractor():
    mesh = Mesh(":memory:")
    ext = LLMExtractor(mock_mode=True, mock_response=TEDX_PAYLOAD)
    mem = mesh.ingest_text("whatever", extractor=ext)
    assert mem.hyperedge.type == "Decision"
    # The persisted hyperedge should carry the participant roles.
    edges = mesh.store.edges_of_type("Decision")
    assert edges and edges[0].arity == 5
    mesh.close()


def test_mesh_ingest_text_falls_back_when_no_bedrock(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    assert bedrock_available() is False
    ext = choose_extractor()
    assert isinstance(ext, HeuristicExtractor)
    mesh = Mesh(":memory:")
    mem = mesh.ingest_text("Eli met David about TEDx.", use_llm=False)
    assert mem.source_text.startswith("Eli met David")
    assert mesh.stats()["hyperedges"] >= 1
    mesh.close()


# --------------------------------------------------------------------------- #
# Extras: selection + tool schema wiring
# --------------------------------------------------------------------------- #


def test_choose_extractor_mock_mode_returns_llm():
    ext = choose_extractor(mock_mode=True)
    assert isinstance(ext, LLMExtractor)
    assert ext.mock_mode is True


def test_choose_extractor_use_llm_true():
    ext = choose_extractor(use_llm=True)
    assert isinstance(ext, LLMExtractor)


def test_record_memory_tool_schema_shape():
    assert RECORD_MEMORY_TOOL["name"] == "record_memory"
    props = RECORD_MEMORY_TOOL["inputSchema"]["json"]["properties"]
    assert set(props) >= {"entities", "hyperedge", "contradictions", "supersedes"}


def test_build_request_forces_tool_choice():
    ext = LLMExtractor(model="global.anthropic.claude-opus-4-8")
    req = ext._build_request("Eli met David", participants=["Eli"], context={"topic": "x"})
    assert req["modelId"] == "global.anthropic.claude-opus-4-8"
    assert req["toolConfig"]["toolChoice"] == {"tool": {"name": "record_memory"}}
    assert req["inferenceConfig"]["maxTokens"] == ext.max_tokens


def test_heuristic_extractor_backend_label():
    assert HeuristicExtractor().backend == "heuristic"
    assert LLMExtractor().backend == "llm"


def test_confidence_clamped_in_lowering():
    payload = {
        "entities": [{"name": "Eli", "kind": "person", "confidence": 5.0}],
        "hyperedge": {
            "type": "Observation",
            "participants": [{"entity_name": "Eli", "role": "subject"}],
            "timestamp": None,
            "confidence": 2.0,
        },
    }
    mem = ExtractedMemory.from_tool_input(payload, source_text="Eli exists.")
    assert mem.entities[0].confidence == 1.0  # clamped
    ex = mem.to_extraction()
    assert 0.0 <= ex.nodes[0].confidence <= 1.0
