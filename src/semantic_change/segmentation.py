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
    """Split the tokens of one sentence before each coordinated clause with its own subject.

    The rule reads the labels that the spaCy parser gives to each token, so the split
    follows the grammar and not the punctuation. The function uses these public spaCy
    ``Token`` attributes: ``i`` (position in the document), ``dep_`` (the grammar role),
    ``children`` (the tokens that depend on this token), ``left_edge`` (the first token
    of the phrase that this token governs) and ``lower_`` (the lowercase text).

    A coordinated verb (``dep_ == "conj"``) that has its own subject starts a new clause.
    A coordinated verb that shares the subject of the first verb does not.

    Example, a sentence that is split: "Users can cancel their plan, but the fee is not
    refunded."

        i   token     dep_
        0   Users     nsubj
        1   can       aux
        2   cancel    ROOT
        3   their     poss
        4   plan      dobj
        5   ,         punct
        6   but       cc
        7   the       det
        8   fee       nsubjpass     <- the subject of "refunded"
        9   is        auxpass
        10  not       neg
        11  refunded  conj          <- a coordinated verb with its own subject
        12  .         punct

    1. ``first`` is 0, the position of the first token.
    2. Find the cuts. Only "refunded" is a ``conj`` with a subject child ("fee"). The cut
       goes where its clause starts, ``left_edge`` = "the" at position 7. A cut at
       ``first`` is removed, because it would only make an empty group. ``cuts == [7]``.
    3. Walk the tokens from left to right and start a new group at position 7:
       ``["Users can cancel their plan , but"]`` and ``["the fee is not refunded ."]``.
    4. "but" is the last token of the first group, but it belongs to the second clause.
       A coordinator at the end of a group moves to the start of the next group:
       ``["Users can cancel their plan ,"]`` and ``["but the fee is not refunded ."]``.
    5. Empty groups are removed, and the groups are returned.

    Example, a sentence that is not split: "Users can cancel and receive a refund."
    "receive" is a ``conj``, but it has no subject of its own: both verbs share "Users".
    ``cuts`` is empty, so all tokens stay in one group.

    The parser decides the labels, so it can miss a clause. In "Users can cancel, the fee
    is refunded, and the account is closed." the parser labels "cancel" as ``ccomp`` and
    not as ``conj``, so only the last clause is cut off and the first two stay together.

    Args:
        tokens: The tokens of one sentence, without whitespace tokens, in text order.

    Returns:
        The groups of tokens, one for each clause. There are no empty groups.
    """
    # The comments below follow this sentence. The numbers are the token positions (``i``):
    #
    #   0     1   2      3     4    5 6   7   8   9  10  11       12
    #   Users can cancel their plan , but the fee is not refunded .
    #
    # The parser labels "refunded" as ``conj`` and "fee" as its subject (``nsubjpass``).

    # Step 1: find the positions where a new clause starts.
    first = tokens[0].i  # position of the first token of the sentence: 0
    cuts = sorted(
        # A token gives a cut if it is a coordinated verb (conj) with its own subject.
        # "refunded" does: its subject is "fee". In "Users can cancel and receive a refund"
        # "receive" does not, because both verbs share the subject "Users".
        #
        # left_edge is the first token of the clause of that verb. The cut goes before the
        # subject "the" (7), not before the verb "refunded" (11). The braces make a set, so
        # two verbs with the same clause start give one cut.
        {
            t.left_edge.i
            for t in tokens
            if t.dep_ == "conj" and any(child.dep_ in SUBJECTS for child in t.children)
        }
        # A cut at the first token would only give an empty first group.
        - {first}
    )  # sorted() puts the cuts in text order. Example: cuts == [7]. Two cuts give 3 groups.

    # Step 2: walk through the tokens once and start a new group at each cut.
    # ``groups`` is a list of clauses, and each clause is a list of tokens. One group is open.
    groups: list[list[Any]] = [[]]
    # ``cut`` is the next position where a new group starts, and None when no cut is left.
    # ``pending`` gives the cuts one by one: cut = 7 now, and None after position 7.
    pending = iter(cuts)
    cut = next(pending, None)
    for token in tokens:
        # Start a new group if the token has reached the next cut. "cut is not None" stops a
        # comparison with None, and "groups[-1]" stops a new group after an empty group.
        if cut is not None and token.i >= cut and groups[-1]:
            groups.append([])
            cut = next(pending, None)  # now wait for the cut after this one
        groups[-1].append(token)  # each token goes into the newest group
    # Result of step 2: [Users can cancel their plan , but] [the fee is not refunded .]
    # The word "but" is in the wrong group: it belongs to the clause that it introduces.

    # Step 3: move a coordinator from the end of a group to the start of the next group.
    # "Users can cancel, but | the fee ..." becomes "Users can cancel, | but the fee ...".
    for k in range(len(groups) - 1):  # every group except the last one has a next group
        # "while", not "if": "Fees apply, and so | users pay" moves "so" and then "and".
        # pop() takes the last token of group k, and insert(0, ...) puts it at the start of
        # group k + 1. The tokens that move keep their order: [and so users pay .].
        while groups[k] and groups[k][-1].lower_ in COORDINATORS:
            groups[k + 1].insert(0, groups[k].pop())
    # Result of step 3: [Users can cancel their plan ,] [but the fee is not refunded .]

    return [group for group in groups if group]  # leave out a group that is empty
