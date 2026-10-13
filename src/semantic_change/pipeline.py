"""The pipeline: it runs the five stages for one pair of texts.

    text A --+
             +--> 1. segment --> 2. embed --> 3. align --+--> 4. agent --+--> 5. output
    text B --+                                           |               |
                                                         +---------------+
                                                     no flagged unit: no Claude call

Only the flagged units go to the agent. If no wording changed, the pipeline gives the
verdict without a Claude call.
"""

from __future__ import annotations

from semantic_change.agent import AgentOutcome, ChangeAgent
from semantic_change.alignment import AlignedUnit, align
from semantic_change.config import Settings
from semantic_change.embeddings import SentenceEmbedder
from semantic_change.models import AgentTrace, Analysis, ChangeResult, PipelineInfo, UnitView
from semantic_change.report import render_markdown
from semantic_change.segmentation import ClauseSegmenter
from semantic_change.taxonomy import ChangeType, Severity


class SemanticChangePipeline:
    """Runs stages 1 to 5. One instance serves all requests.

    The keyword arguments replace the default components, for example in tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        segmenter: ClauseSegmenter | None = None,
        embedder: SentenceEmbedder | None = None,
        agent: ChangeAgent | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.segmenter = segmenter or ClauseSegmenter(self.settings.spacy_model)
        self.embedder = embedder or SentenceEmbedder(self.settings.embedding_model)
        self.agent = agent or ChangeAgent(self.settings)

    def warmup(self) -> None:
        """Load the spaCy and embedding models, so the first request is not slow."""
        self.segmenter.load()
        self.embedder.load()

    def detect(self, text_a: str, text_b: str) -> ChangeResult:
        """Return only the verdict, in the output format of the assignment."""
        return self.analyze(text_a, text_b).result

    def analyze(self, text_a: str, text_b: str) -> Analysis:
        """Return the verdict, the changed units, the agent trace and the Markdown report.

        Raises:
            ValueError: A text is longer than ``max_text_chars``.
            AgentError: The evaluation agent gave no verdict.
        """
        for name, text in (("text_a", text_a), ("text_b", text_b)):
            if len(text) > self.settings.max_text_chars:
                raise ValueError(f"{name} is longer than {self.settings.max_text_chars} characters")

        units_a = self.segmenter.segment(text_a)
        units_b = self.segmenter.segment(text_b)
        aligned = align(units_a, units_b, self.embedder, self.settings.align_threshold)
        flagged = [u for u in aligned if u.flagged]
        info = PipelineInfo(units_a=len(units_a), units_b=len(units_b), flagged_units=len(flagged))
        if flagged:
            outcome = self.agent.run(aligned, info)
        else:
            outcome = AgentOutcome(_unchanged(aligned), {}, AgentTrace(model=self.agent.model))

        # The response leaves out the units with the status "unchanged".
        units = [_view(u, outcome) for u in aligned if u.status != "unchanged"]
        report = render_markdown(outcome.verdict, units)
        return Analysis(
            result=outcome.verdict, units=units, pipeline=info, agent=outcome.trace, report=report
        )


def _unchanged(aligned: list[AlignedUnit]) -> ChangeResult:
    """The verdict when no wording changed: a new sentence order, or no change at all."""
    if any(u.status == "moved" for u in aligned):
        return ChangeResult(
            change_detected=False,
            change_type=ChangeType.PARAPHRASE,
            summary="The same statements appear in a different order.",
            severity=Severity.LOW,
            confidence=0.9,
            evidence=[],
        )
    return ChangeResult(
        change_detected=False,
        change_type=ChangeType.NO_CHANGE,
        summary="The texts are identical apart from case, spacing or punctuation.",
        severity=Severity.LOW,
        confidence=0.99,
        evidence=[],
    )


def _view(unit: AlignedUnit, outcome: AgentOutcome) -> UnitView:
    """The response view of one aligned unit, with the finding of the agent."""
    return UnitView(
        id=unit.id,
        status=unit.status,
        similarity=unit.similarity,
        moved=unit.moved,
        original=unit.original_text,
        modified=unit.modified_text,
        finding=outcome.findings.get(unit.id),
    )
