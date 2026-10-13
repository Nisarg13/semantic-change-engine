"""Shared test helpers.

- ``segmenter`` and ``embedder``: the real models of stages 1 and 2, loaded once.
- ``ScriptedClient``: replaces the Anthropic client. It returns scripted replies and
  records each request, so the tests make no network calls and need no API key.
- ``unit`` and ``aligned``: hand-made units for the agent tests.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_change.alignment import AlignedUnit
from semantic_change.embeddings import SentenceEmbedder
from semantic_change.segmentation import ClauseSegmenter, Unit


@pytest.fixture(scope="session")
def segmenter() -> ClauseSegmenter:
    return ClauseSegmenter()


@pytest.fixture(scope="session")
def embedder() -> SentenceEmbedder:
    return SentenceEmbedder()


# ------------------------------------------------------------- scripted Claude
def tool(name: str, call_id: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def reply(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=800,
        output_tokens=200,
    )  # fmt: skip
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage)


def verdict(**overrides: Any) -> dict[str, Any]:
    values = {
        "change_detected": True,
        "change_type": "restriction_added",
        "severity": "high",
        "confidence": 0.85,
        "summary": "Cancellation changed from anytime to a 7-day limit.",
        "evidence": ["anytime → within 7 days"],
    }
    return {**values, **overrides}


def finding(unit_id: int, **overrides: Any) -> dict[str, Any]:
    risk = "Subscribers lose the right to cancel after 7 days."
    return {"unit_id": unit_id, **verdict(**overrides), "risk": overrides.get("risk", risk)}


class ScriptedClient:
    """Stands in for anthropic.Anthropic: returns scripted replies, records requests."""

    def __init__(self, *replies: SimpleNamespace, error: Exception | None = None) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(copy.deepcopy(request))
        if self.error is not None:
            raise self.error
        if not self.replies:
            raise AssertionError("the agent made an unexpected Claude call")
        return self.replies.pop(0)


# ------------------------------------------------------------ hand-made units
def unit(text_value: str, index: int = 0) -> Unit:
    tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text_value)
    return Unit(
        index=index,
        text=text_value,
        start=0,
        end=len(text_value),
        tokens=tuple(tokens),
        spaces=tuple(True for _ in tokens),
        punct=tuple(not re.match(r"\w", t) for t in tokens),
    )


def aligned(uid: int, original: str, modified: str, status: str = "modified") -> AlignedUnit:
    return AlignedUnit(
        id=uid,
        status=status,  # type: ignore[arg-type]
        original=(unit(original),) if original else (),
        modified=(unit(modified),) if modified else (),
        similarity=0.0 if status in ("inserted", "deleted") else 0.9,
    )
