"""Request and response schemas of the REST API.

FastAPI uses these pydantic models to validate the requests and to build the OpenAPI
documentation at ``/docs``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semantic_change.taxonomy import ChangeType, Severity

# Result of stage 3 for one aligned unit.
UnitStatus = Literal["unchanged", "moved", "modified", "inserted", "deleted"]


class ChangeRequest(BaseModel):
    """Request body of ``/v1/detect`` and ``/v1/analyze``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "text_a": "Users can cancel their subscription anytime.",
                    "text_b": "Users can cancel their subscription within 7 days.",
                }
            ]
        },
    )

    text_a: str = Field(description="Original text.")
    text_b: str = Field(description="Modified text.")


class ChangeResult(BaseModel):
    """Verdict for the whole document, in the output format of the assignment."""

    model_config = ConfigDict(extra="forbid")

    change_detected: bool = Field(description="True if the meaning changed.")
    change_type: ChangeType = Field(description="Category of the most significant change.")
    summary: str = Field(description="Short explanation of the change in meaning.")
    severity: Severity = Field(description="Impact of the change: low, medium or high.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the verdict, from 0 to 1.")
    evidence: list[str] = Field(description="Key phrase changes, e.g. 'anytime → within 7 days'.")


class Finding(ChangeResult):
    """Verdict of the agent for one flagged unit."""

    risk: str = Field(description="Who is affected, and the legal, financial or system risk.")


class UnitView(BaseModel):
    """One aligned unit that is different in text B: modified, inserted, deleted or moved."""

    id: int = Field(description="Position of the unit in the alignment, from 1.")
    status: UnitStatus
    similarity: float = Field(description="Cosine similarity of the two texts, from 0 to 1.")
    moved: bool = Field(description="True if the unit changed its position.")
    original: str = Field(description="The unit in text A. Empty for an inserted unit.")
    modified: str = Field(description="The unit in text B. Empty for a deleted unit.")
    finding: Finding | None = Field(
        default=None, description="Verdict of the agent. Null for a moved unit."
    )


class PipelineInfo(BaseModel):
    """Unit counts of stages 1 to 3."""

    units_a: int = Field(description="Units in text A.")
    units_b: int = Field(description="Units in text B.")
    flagged_units: int = Field(description="Units sent to the evaluation agent.")


class AgentTrace(BaseModel):
    """Work of the evaluation agent for one request."""

    model: str = Field(description="Claude model ID.")
    turns: int = Field(default=0, description="Claude calls. 0 if no wording changed.")
    tool_calls: int = Field(default=0, description="Tool calls that Claude made.")
    input_tokens: int = Field(default=0, description="All input tokens, cached or not.")
    output_tokens: int = Field(default=0, description="Output tokens, thinking included.")


class Analysis(BaseModel):
    """Response of ``/v1/analyze``."""

    result: ChangeResult = Field(description="The verdict, the same as from /v1/detect.")
    units: list[UnitView] = Field(description="The units that are different in text B.")
    pipeline: PipelineInfo
    agent: AgentTrace
    report: str = Field(description="Markdown report for human reviewers.")
