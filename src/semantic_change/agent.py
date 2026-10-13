"""Stage 4 - LLM evaluation loop: a Claude agent that judges each flagged unit.

The agent works in a tool loop, and Claude decides which tools to call:

- ``inspect_unit`` returns the word edits of a unit and the text around it.
- ``record_finding`` records the verdict and the risk for one flagged unit.
- ``submit_verdict`` records the verdict for the whole document and ends the loop.

The code validates each tool input. An input that is not valid goes back to Claude as an
``is_error`` result, so Claude can correct the call. ``submit_verdict`` fails until each
flagged unit has a finding.

Request settings: adaptive thinking, an effort level, ``strict`` tool schemas and prompt
caching. ``tool_choice`` stays ``auto``, because Claude Opus 5.5 and Sonnet 5.5 reject
forced tool use. The conversation is append-only: each reply goes back to the API
unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import ValidationError

from semantic_change.alignment import AlignedUnit
from semantic_change.config import Settings
from semantic_change.models import AgentTrace, ChangeResult, Finding, PipelineInfo
from semantic_change.prompts import SYSTEM_PROMPT, build_task
from semantic_change.taxonomy import ChangeType, Severity


class AgentError(RuntimeError):
    """The agent gave no verdict: an API error, a refusal or the turn limit."""


class LLMNotConfiguredError(AgentError):
    """The Claude API key is missing or not valid."""


# JSON schema properties of a verdict, shared by record_finding and submit_verdict.
_VERDICT_PROPERTIES: dict[str, Any] = {
    "change_detected": {"type": "boolean", "description": "True if the meaning changed."},
    "change_type": {"type": "string", "enum": [c.value for c in ChangeType]},
    "severity": {"type": "string", "enum": [s.value for s in Severity]},
    "confidence": {"type": "number", "description": "Probability 0-1 that the category is right."},
    "summary": {"type": "string", "description": "One or two plain sentences."},
    "evidence": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Key phrase changes as '<A phrase> → <B phrase>'; empty if no change.",
    },
}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    """A client tool definition with a strict schema. All properties are required."""
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "inspect_unit",
        "Return the word-level edits of one aligned unit and the text around it. Call it "
        "when you need the exact edits or context to resolve a reference.",
        {"unit_id": {"type": "integer", "description": "Id of an aligned unit."}},
    ),
    _tool(
        "record_finding",
        "Record your verdict for one flagged unit. Call it once for every flagged unit, "
        "also when the meaning did not change.",
        {
            "unit_id": {"type": "integer", "description": "Id of the flagged unit."},
            **_VERDICT_PROPERTIES,
            "risk": {
                "type": "string",
                "description": "Who is affected and what legal, financial, compliance, "
                "security or operational risk it creates; empty if no change.",
            },
        },
    ),
    _tool(
        "submit_verdict",
        "Submit the verdict for the whole document, after every flagged unit has a "
        "finding. Call it exactly once; it ends the review.",
        _VERDICT_PROPERTIES,
    ),
]


@dataclass
class AgentOutcome:
    """Result of the agent: the document verdict, the findings by unit id and the trace."""

    verdict: ChangeResult
    findings: dict[int, Finding]
    trace: AgentTrace


@dataclass
class _Review:
    """State of one review: the units, the findings so far and the verdict."""

    units: dict[int, AlignedUnit]
    flagged: list[int]
    findings: dict[int, Finding] = field(default_factory=dict)
    verdict: ChangeResult | None = None

    @property
    def missing(self) -> list[int]:
        """The flagged units that have no finding yet."""
        return [i for i in self.flagged if i not in self.findings]


class ChangeAgent:
    """Runs the evaluation loop with the Claude Messages API.

    ``client`` replaces the Anthropic client, for example with a scripted client in tests.
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or Settings()
        self._client = client

    @property
    def model(self) -> str:
        """The Claude model ID."""
        return self.settings.llm_model

    def run(self, units: list[AlignedUnit], info: PipelineInfo) -> AgentOutcome:
        """Judge the flagged units and return the verdict.

        Args:
            units: All aligned units of stage 3. The agent judges the flagged units and
                uses the others as context.
            info: The unit counts for the task message.

        Raises:
            LLMNotConfiguredError: The API key is missing or not valid.
            AgentError: Claude refused, the reply was cut off, the API failed, or no
                verdict came within ``agent_max_turns`` turns.
        """
        review = _Review(units={u.id: u for u in units}, flagged=[u.id for u in units if u.flagged])
        trace = AgentTrace(model=self.model)
        messages: list[dict[str, Any]] = [{"role": "user", "content": build_task(units, info)}]
        for turn in range(1, self.settings.agent_max_turns + 1):
            response = self._create(messages)
            trace.turns = turn
            usage = response.usage  # input_tokens excludes the tokens written to or read from cache
            trace.input_tokens += (
                usage.input_tokens
                + (usage.cache_creation_input_tokens or 0)
                + (usage.cache_read_input_tokens or 0)
            )
            trace.output_tokens += usage.output_tokens
            # A refused or cut-off reply can hold incomplete tool calls: run none of them.
            if response.stop_reason in ("refusal", "max_tokens"):
                raise AgentError(f"Claude stopped early (stop_reason={response.stop_reason})")
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            trace.tool_calls += len(calls)
            results = [self._execute(call, review) for call in calls]
            if review.verdict is not None:
                return AgentOutcome(review.verdict, review.findings, trace)
            # A reply without tool calls gets a reminder to use the tools.
            messages.append({"role": "user", "content": results or _nudge(review)})
        raise AgentError(f"The agent gave no verdict in {self.settings.agent_max_turns} turns")

    # ------------------------------------------------------------------ tools
    def _execute(self, call: Any, review: _Review) -> dict[str, Any]:
        """Run one tool call and return its tool_result block. Errors go back to Claude."""
        try:
            content = self._dispatch(call.name, dict(call.input), review)
            return {"type": "tool_result", "tool_use_id": call.id, "content": content}
        except (ValueError, ValidationError, KeyError, TypeError) as exc:
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }

    def _dispatch(self, name: str, args: dict[str, Any], review: _Review) -> str:
        """Run the tool ``name`` and return its result text for Claude."""
        if name == "inspect_unit":
            return json.dumps(_inspect(review, _unit_id(args, review)), ensure_ascii=False)
        if name == "record_finding":
            unit_id = _unit_id(args, review, flagged_only=True)
            review.findings[unit_id] = Finding.model_validate(normalize(args, "unit_id"))
            left = review.missing
            return f"Recorded unit {unit_id}." + (
                f" Still without a finding: {left}." if left else " All flagged units are done."
            )
        if name == "submit_verdict":
            if review.missing:
                raise ValueError(f"record a finding for units {review.missing} first")
            review.verdict = ChangeResult.model_validate(normalize(args))
            return "Verdict submitted."
        raise ValueError(f"unknown tool '{name}'")

    # -------------------------------------------------------------------- API
    def _create(self, messages: list[dict[str, Any]]) -> Any:
        """Send one request to the Claude Messages API and return the reply."""
        try:
            return self._get_client().messages.create(
                model=self.model,
                max_tokens=self.settings.llm_max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.llm_effort},
                cache_control={"type": "ephemeral"},  # cache the prompt for the next turn
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMNotConfiguredError("Claude rejected ANTHROPIC_API_KEY") from exc
        except anthropic.APIError as exc:
            raise AgentError(f"Claude API error: {exc}") from exc
        except TypeError as exc:  # the SDK raises TypeError when it finds no API key
            if "authentication" not in str(exc).lower():
                raise
            raise LLMNotConfiguredError("No Claude credentials: set ANTHROPIC_API_KEY") from exc

    def _get_client(self) -> Any:
        """Create the Anthropic client on first use. It reads ANTHROPIC_API_KEY."""
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client


def normalize(args: dict[str, Any], *drop: str) -> dict[str, Any]:
    """Make a verdict agree with its change type, and remove the keys in ``drop``.

    ``change_detected`` follows from ``change_type``. If the meaning did not change, the
    severity is low, and the evidence and the risk are empty.
    """
    data = {k: v for k, v in args.items() if k not in drop}
    data["change_detected"] = ChangeType(data["change_type"]).is_meaningful
    if not data["change_detected"]:
        data.update(severity=Severity.LOW.value, evidence=[])
        if "risk" in data:
            data["risk"] = ""
    return data


def _unit_id(args: dict[str, Any], review: _Review, flagged_only: bool = False) -> int:
    """The unit id of a tool input. Raises ValueError if the id is not valid."""
    unit_id = args.get("unit_id")
    if not isinstance(unit_id, int) or unit_id not in review.units:
        raise ValueError(f"unit {unit_id} does not exist; valid ids: {sorted(review.units)}")
    if flagged_only and unit_id not in review.flagged:
        raise ValueError(f"unit {unit_id} is not flagged; flagged units: {review.flagged}")
    return unit_id


def _inspect(review: _Review, unit_id: int) -> dict[str, Any]:
    """Result of ``inspect_unit``: the unit, its word edits and the text around it."""
    unit = review.units[unit_id]
    before, after = review.units.get(unit_id - 1), review.units.get(unit_id + 1)
    return {
        "unit_id": unit_id,
        "status": unit.status,
        "similarity": unit.similarity,
        "moved": unit.moved,
        "original": unit.original_text,
        "modified": unit.modified_text,
        "word_edits": unit.word_edits(),
        "text_before": (before.modified_text or before.original_text) if before else "",
        "text_after": (after.modified_text or after.original_text) if after else "",
    }


def _nudge(review: _Review) -> str:
    """The reminder for Claude after a reply without tool calls."""
    if review.missing:
        return f"Record a finding for units {review.missing}, then call submit_verdict."
    return "Every flagged unit has a finding. Call submit_verdict now."
