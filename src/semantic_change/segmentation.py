"""Stage 1 - granular segmentation: split a text into clause units.

spaCy's English pipeline parses each sentence into a dependency tree. spaCy has no
constituency parser, so the clause rule uses dependency labels: a coordinated verb
(``conj``) that has its own subject starts a new unit.

    "Users can cancel their plan," | "but the setup fee is not refunded."

A subordinate clause ("if ...", "unless ...", "who ...") stays in the unit of the clause
that it qualifies, because it changes the meaning of that clause. A line break always
ends a sentence, so each item of a bullet list is a unit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Dependency labels of a clause subject.
SUBJECTS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"})
# Words that join two clauses. At a split, the word moves to the start of the next unit.
COORDINATORS = frozenset({"but", "and", "or", "yet", "so", "however", "whereas", "while"})


@Language.component("scd_line_breaks")
def _line_breaks(doc: Doc) -> Doc:
    """spaCy component: start a new sentence after each line break."""
    for token in doc[:-1]:
        if token.is_space and "\n" in token.text:
            doc[token.i + 1].is_sent_start = True
    return doc


@dataclass(frozen=True, slots=True)
class Unit:
    """One clause of a text, with its position and tokens."""

    index: int  # position in the text, from 0
    text: str  # the clause as it occurs in the text
    start: int  # character offset of the first character
    end: int  # character offset after the last character
    tokens: tuple[str, ...]  # token texts in their original case, without whitespace
    spaces: tuple[bool, ...]  # True if whitespace follows the token
    punct: tuple[bool, ...]  # True if the token is punctuation

    @property
    def key(self) -> tuple[str, ...]:
        """The lowercase words without punctuation. Equal keys mean equal wording."""
        return tuple(t.lower() for t, p in zip(self.tokens, self.punct, strict=True) if not p)


class ClauseSegmenter:
    """Splits text into clause units with a spaCy pipeline. Threads can share one instance."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.model = model
        self._nlp: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the spaCy pipeline on the first call. NER and the lemmatizer are not used."""
        with self._lock:
            if self._nlp is None:
                nlp = spacy.load(self.model, exclude=["ner", "lemmatizer"])
                nlp.add_pipe("scd_line_breaks", before="parser")
                self._nlp = nlp

    def segment(self, text: str) -> list[Unit]:
        """Split ``text`` into clause units, in text order. A blank text gives no units."""
        if not text.strip():
            return []
        self.load()
        with self._lock:  # the API handles requests in worker threads: one parse at a time
            doc = self._nlp(text)
        units: list[Unit] = []
        pending: list[Any] = []  # a bullet ("-") that the parser made into its own sentence
        for sentence in doc.sents:
            tokens = pending + [t for t in sentence if not t.is_space]
            if not any(not t.is_punct for t in tokens):
                pending = tokens  # punctuation only: put it in front of the next sentence
                continue
            pending = []
            for group in _clauses(tokens):
                if all(t.is_punct for t in group):
                    continue
                start, end = group[0].idx, group[-1].idx + len(group[-1].text)
                units.append(
                    Unit(
                        index=len(units),
                        text=text[start:end],
                        start=start,
                        end=end,
                        tokens=tuple(t.text for t in group),
                        spaces=tuple(bool(t.whitespace_) for t in group),
                        punct=tuple(t.is_punct for t in group),
                    )
                )
        return units


def _clauses(tokens: list[Any]) -> list[list[Any]]:
    """Split the tokens of one sentence before each coordinated clause with its own subject."""
    first = tokens[0].i
    cuts = sorted(
        {
            t.left_edge.i
            for t in tokens
            if t.dep_ == "conj" and any(child.dep_ in SUBJECTS for child in t.children)
        }
        - {first}
    )
    groups: list[list[Any]] = [[]]
    pending = iter(cuts)
    cut = next(pending, None)
    for token in tokens:
        if cut is not None and token.i >= cut and groups[-1]:
            groups.append([])
            cut = next(pending, None)
        groups[-1].append(token)
    # "Users can cancel, but | the fee ..." becomes "Users can cancel, | but the fee ...".
    for k in range(len(groups) - 1):
        while groups[k] and groups[k][-1].lower_ in COORDINATORS:
            groups[k + 1].insert(0, groups[k].pop())
    return [group for group in groups if group]
