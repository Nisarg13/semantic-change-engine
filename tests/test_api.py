"""REST API end to end: real segmentation, embeddings and alignment; scripted Claude."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semantic_change.agent import ChangeAgent
from semantic_change.api import create_app
from semantic_change.config import Settings
from semantic_change.embeddings import SentenceEmbedder
from semantic_change.pipeline import SemanticChangePipeline
from semantic_change.segmentation import ClauseSegmenter

from .conftest import ScriptedClient, finding, reply, tool, verdict

EXAMPLE = {
    "text_a": "Users can cancel their subscription anytime.",
    "text_b": "Users can cancel their subscription within 7 days.",
}
CONTRACT = {"change_detected", "change_type", "summary", "severity", "confidence", "evidence"}


@pytest.fixture
def make_client(segmenter: ClauseSegmenter, embedder: SentenceEmbedder):
    def build(claude: ScriptedClient, **settings) -> TestClient:
        config = Settings(**settings)
        pipeline = SemanticChangePipeline(
            config, segmenter=segmenter, embedder=embedder, agent=ChangeAgent(config, claude)
        )
        return TestClient(create_app(pipeline))

    return build


def one_review() -> ScriptedClient:
    return ScriptedClient(
        reply(tool("record_finding", "t1", **finding(1)), tool("submit_verdict", "t2", **verdict()))
    )


def test_detect_returns_the_assignment_format(make_client) -> None:
    response = make_client(one_review()).post("/v1/detect", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == CONTRACT
    assert body["change_type"] == "restriction_added"
    assert body["evidence"] == ["anytime → within 7 days"]


def test_analyze_returns_units_trace_and_report(make_client) -> None:
    body = make_client(one_review()).post("/v1/analyze", json=EXAMPLE).json()
    assert body["pipeline"] == {"units_a": 1, "units_b": 1, "flagged_units": 1}
    [unit] = body["units"]
    assert unit["status"] == "modified"
    assert unit["finding"]["risk"].startswith("Subscribers")
    assert body["agent"]["turns"] == 1
    assert body["report"].startswith("# Semantic change report")
    assert "## Changes (1)" in body["report"]
    assert "**Why it matters:**" in body["report"]


@pytest.mark.parametrize(
    ("text_b", "change_type"),
    [
        ("users can cancel their subscription anytime", "no_change"),
        ("", None),
    ],
)
def test_unchanged_wording_needs_no_llm_call(make_client, text_b: str, change_type) -> None:
    claude = ScriptedClient()  # fails the test if the agent calls Claude
    if change_type is None:
        response = make_client(claude).post("/v1/detect", json={"text_a": "", "text_b": text_b})
        assert response.json()["change_type"] == "no_change"
    else:
        response = make_client(claude).post("/v1/detect", json={**EXAMPLE, "text_b": text_b})
        assert response.json()["change_type"] == change_type
    assert claude.requests == []


def test_reordered_sentences_are_a_paraphrase_without_llm(make_client) -> None:
    claude = ScriptedClient()
    body = make_client(claude).post(
        "/v1/analyze",
        json={"text_a": "Orders ship daily. Returns are free.",
              "text_b": "Returns are free. Orders ship daily."},
    ).json()  # fmt: skip
    assert body["result"]["change_type"] == "paraphrase"
    assert [u["status"] for u in body["units"]] == ["moved"]
    assert body["agent"]["turns"] == 0
    assert claude.requests == []


def test_missing_credentials_is_503(make_client) -> None:
    claude = ScriptedClient(error=TypeError("Could not resolve authentication method"))
    response = make_client(claude).post("/v1/detect", json=EXAMPLE)
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_agent_failure_is_502(make_client) -> None:
    claude = ScriptedClient(reply(tool("submit_verdict", "t1", **verdict()), stop="refusal"))
    assert make_client(claude).post("/v1/detect", json=EXAMPLE).status_code == 502


def test_invalid_input_is_422(make_client) -> None:
    client = make_client(ScriptedClient(), max_text_chars=20)
    assert client.post("/v1/detect", json={"text_a": "x" * 21, "text_b": "y"}).status_code == 422
    assert client.post("/v1/detect", json={"text_a": "only one"}).status_code == 422


def test_health(make_client) -> None:
    body = make_client(ScriptedClient()).get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_model"] == "claude-sonnet-5-5"
