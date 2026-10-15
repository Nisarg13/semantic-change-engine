"""Stage 3 - semantic alignment and clustering: match the units of text A and text B.

1. Units with the same wording match first, in order (``difflib``). They need no
   embedding, and a repeated sentence keeps its position.
2. The other units form a bipartite graph. The weight of an edge is the cosine
   similarity of the two units minus a small penalty for a change of position. The
   Hungarian algorithm (SciPy ``linear_sum_assignment``) finds the matching with the
   highest total weight, so a sentence that moved still matches its counterpart. A pair
   below the threshold is not a match: its units become a deletion and an insertion.
3. Clustering: an unmatched unit joins a neighboring match when the joined text is more
   similar to the counterpart. This step finds a sentence that was split in two and two
   sentences that were joined.

Example. Text A and text B have four units each:

    A0 Refunds are available within 30 days.    B0 Shipping is free for all orders.
    A1 Shipping is free for all orders.         B1 Refunds are available within 14 days.
    A2 Support answers questions by email.      B2 Support answers questions by email and chat.
    A3 Data is stored in the EU.                B3 Cookies track page visits.

    Step 1: A1 and B0 have the same wording, so they match (similarity 1.0).
    Step 2: A0 matches B1 (0.93) and A2 matches B2 (0.87). The best pair that is left, A3
            and B3, scores 0.22. That is below the threshold, so it is not a match.
    Step 3: No lonely unit makes a match more similar, so nothing joins.
    Result: five aligned units in the order of text B: 1 unchanged (shipping), 2 modified
            (refunds, and it moved), 3 modified (support), 4 inserted (cookies) and
            5 deleted (EU data). Only the units 2 to 5 are sent to Claude.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from semantic_change.models import UnitStatus
from semantic_change.segmentation import Unit

# Weight of the position penalty. Positions are scaled to the range 0-1 (0 is the first unit
# of a text, 1 is the last). A pair of units at the two ends loses 0.05 of its similarity, and
# a pair at the same position loses nothing. The penalty is small: it only decides between
# matches that are almost equally good.
POSITION_WEIGHT = 0.05
# Minimum increase of similarity for an unmatched unit to join a neighboring match.
# Example: a match that goes from 0.75 to 0.97 with the new unit joins (gain 0.22), and a
# match that goes from 0.87 to 0.57 does not.
MERGE_GAIN = 0.05


class Embedder(Protocol):
    """Encodes texts to normalized vectors, for example ``SentenceEmbedder``.

    ``align`` needs only the method ``encode``, so a test can pass a simple fake instead of
    the real model. ``encode`` returns an array with one row for each text. With the default
    MiniLM model, a row has 384 numbers. A row has the length 1, so the dot product of two
    rows is their cosine similarity.
    """

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class AlignedUnit:
    """A group of corresponding units, usually one unit of text A and one of text B.

    An inserted unit has no original units. A deleted unit has no modified units.

    The status says what happened to the unit:

        unchanged  the same words at the same place    (not sent to Claude)
        moved      the same words at another place     (not sent to Claude)
        modified   the words changed                   (sent to Claude)
        inserted   the unit is only in text B          (sent to Claude)
        deleted    the unit is only in text A          (sent to Claude)

    Example: "Refunds are available within 30 days." in text A and "Refunds are available
    within 14 days." in text B give one aligned unit with the status ``modified``, the
    similarity 0.93 and ``word_edits() == ["'30' → '14'"]``.

    A unit can also hold several units of one text, for example one sentence of text A that
    text B split in two: ``original`` has one unit and ``modified`` has two.
    """

    id: int  # position in the alignment, from 1
    status: UnitStatus
    original: tuple[Unit, ...]  # units of text A
    modified: tuple[Unit, ...]  # units of text B
    similarity: float  # cosine similarity of the two texts; 1.0 for the same wording
    moved: bool = False  # the unit is not in the same order as in text A

    @property
    def original_text(self) -> str:
        """The text of the original units, joined with a space. Empty for an inserted unit.

        Example: "Refunds are available within 30 days."
        """
        return " ".join(u.text for u in self.original)

    @property
    def modified_text(self) -> str:
        """The text of the modified units, joined with a space. Empty for a deleted unit.

        Example: "Refunds are available within 14 days."
        """
        return " ".join(u.text for u in self.modified)

    @property
    def flagged(self) -> bool:
        """True if the wording changed. The evaluation agent must judge each flagged unit.

        Modified, inserted and deleted units are flagged. Unchanged and moved units have the
        same words, so they cost no Claude call.
        """
        return self.status in ("modified", "inserted", "deleted")

    def word_edits(self) -> list[str]:
        """Word-level edits from the original text to the modified text.

        Examples: "'anytime' → 'within 7 days'", "added 'not'" and "removed 'only'". An
        inserted or deleted unit gives one "added: ..." or "removed: ..." item.

        How it works, for "The fee is $5." and "The fee is $10.":

        1. Both texts are split into tokens (words and punctuation marks), in lower case.
        2. ``SequenceMatcher`` lists what is equal and what is different. Here "5" became
           "10", and the other tokens are equal.
        3. An edit that is only punctuation is skipped, because it does not change the meaning.
        4. ``_whole_words`` widens the edit to whole words. The "$" is attached to the "5", so
           the edit becomes "$5" → "$10".
        5. ``_render`` writes the tokens as the user typed them: "'$5' → '$10'".
        """
        if not self.original:  # an inserted unit: all of its text is new
            return [f"added: {self.modified_text}"]
        if not self.modified:  # a deleted unit: all of its text is gone
            return [f"removed: {self.original_text}"]
        # The tokens of both sides, each with its spacing and its punctuation flag.
        a, b = _tokens(self.original), _tokens(self.modified)
        # Compare the lower-case words, so a change of case alone is not an edit.
        matcher = SequenceMatcher(
            a=[t.lower() for t, _, _ in a], b=[t.lower() for t, _, _ in b], autojunk=False
        )
        edits = []
        # Each opcode (tag, i0, i1, j0, j1) says: "a[i0:i1] became b[j0:j1]". The tag is
        # "equal", "replace", "delete" or "insert".
        for tag, i0, i1, j0, j1 in matcher.get_opcodes():
            if tag == "equal" or all(p for _, _, p in (*a[i0:i1], *b[j0:j1])):
                continue  # a punctuation-only edit does not change the meaning
            # Widen the edit to whole words, for example "5" → "10" becomes "$5" → "$10".
            i0, i1, j0, j1 = _whole_words(a, b, i0, i1, j0, j1)
            old, new = _render(a[i0:i1]), _render(b[j0:j1])  # the tokens as they were typed
            if old and new:  # something was replaced
                edits.append(f"'{old}' → '{new}'")
            elif old:  # something was removed
                edits.append(f"removed '{old}'")
            else:  # something was added
                edits.append(f"added '{new}'")
        return edits


def align(
    units_a: list[Unit],
    units_b: list[Unit],
    embedder: Embedder,
    threshold: float = 0.45,
) -> list[AlignedUnit]:
    """Match the units of text A to the units of text B.

    This answers the question: "which unit of the old text became which unit of the new
    text?". Only the units whose words changed are sent to Claude later, so a good match
    saves money and shows the reviewer what changed.

    The function works in three steps, from the most certain to the least certain:

    1. Same wording. Units with the same words match at once. This needs no vectors.
    2. Best pairing. The units that are left are compared by meaning (the cosine similarity
       of their vectors), with a small penalty for a change of position. The Hungarian
       algorithm finds the pairing with the highest total score. A pair below ``threshold``
       is thrown away.
    3. Clustering. A unit that is still alone joins a neighboring match if that makes the
       match clearly more similar. This finds a sentence that was split or joined.

    Example:

        text A: "Refunds are available within 30 days of purchase. Shipping is free for all
                 orders."
        text B: "Shipping is free for all orders. Refunds are available within 14 days of
                 purchase. A processing fee of $5 applies to all refunds."

    Step 1 matches A1 with B0 (the same words). Step 2 matches A0 with B1 (similarity 0.93).
    B2 has no partner. The result has three aligned units in the order of text B:

        id 1  unchanged  1.00  Shipping is free for all orders.
        id 2  modified   0.93  Refunds ... 30 days ... → Refunds ... 14 days ... (moved)
        id 3  inserted   0.00  A processing fee of $5 applies to all refunds.

    Args:
        units_a: The units of the original text (stage 1).
        units_b: The units of the modified text (stage 1).
        embedder: The encoder of stage 2.
        threshold: The minimum cosine similarity of a match.

    Returns:
        The aligned units in the order of text B. A deleted unit comes before the match
        of the next unit of text A. The ids go from 1.
    """
    # Step 1: match the units that have exactly the same words.
    # ``pairs`` maps the number of a unit of text A to the number of its partner in text B,
    # and ``scores`` maps the same number to the similarity of the pair.
    # Example: pairs == {1: 0} and scores == {1: 1.0} mean "A1 matches B0, similarity 1.0".
    pairs: dict[int, int] = {}
    scores: dict[int, float] = {}
    # ``Unit.key`` is the lower-case words without punctuation, so case and punctuation do
    # not matter. SequenceMatcher is an object that holds the two lists. It finds the runs
    # of equal keys that are in the same order in both texts when we call
    # ``get_matching_blocks``.
    keys = SequenceMatcher(a=[u.key for u in units_a], b=[u.key for u in units_b], autojunk=False)
    # A block Match(a, b, size) says: "from unit a of text A and unit b of text B, ``size``
    # units in a row are equal". The last block always has the size 0 and adds nothing.
    for block in keys.get_matching_blocks():
        for k in range(block.size):  # walk along the run: A[a + k] matches B[b + k]
            pairs[block.a + k] = block.b + k
            scores[block.a + k] = 1.0  # the same words need no vectors: similarity 1.0

    # Step 2: pair the units that are left by meaning (Hungarian matching).
    # Example (30 and 14 days): left_a == [0], used_b == {0} and left_b == [1, 2].
    left_a = [i for i in range(len(units_a)) if i not in pairs]  # units of A without a partner
    used_b = set(pairs.values())  # units of B that have a partner already
    left_b = [j for j in range(len(units_b)) if j not in used_b]  # units of B without a partner
    if left_a and left_b:  # nothing to pair if one side has no unit left
        # One vector (384 numbers) for each unit that is left.
        va = embedder.encode([units_a[i].text for i in left_a])
        vb = embedder.encode([units_b[j].text for j in left_b])
        # A table of cosine similarities: row i is a unit of A and column j is a unit of B.
        # "@" is a matrix product, and "vb.T" turns the rows of vb into columns.
        # Example: [[0.93, 0.58]] means that A0 is much more like B1 than like B2.
        sims = va @ vb.T
        # Where is each unit in its text? 0 is the first unit and 1 is the last. This makes
        # texts of different lengths comparable. ``max(..., 1)`` avoids a division by zero
        # when a text has only one unit.
        pos_a = np.array([i / max(len(units_a) - 1, 1) for i in left_a])
        pos_b = np.array([j / max(len(units_b) - 1, 1) for j in left_b])
        # The weight of a pair is its similarity minus a small penalty for a change of place.
        # ``pos_a[:, None]`` is a column and ``pos_b[None, :]`` is a row, so the subtraction
        # gives a table with the distance between every unit of A and every unit of B.
        # The penalty is at most POSITION_WEIGHT, so it only decides between close matches.
        weights = sims - POSITION_WEIGHT * np.abs(pos_a[:, None] - pos_b[None, :])
        # The Hungarian algorithm chooses one column for each row (no column twice) so that
        # the sum of the chosen weights is the highest. ``maximize`` is needed because the
        # function looks for the smallest sum by default. The pairs are rows[n] with cols[n].
        rows, cols = linear_sum_assignment(weights, maximize=True)
        single = len(units_a) == len(units_b) == 1  # two one-unit texts always correspond
        for r, c in zip(rows, cols, strict=True):
            # Keep the pair only if the units are alike enough. The test uses ``sims``, the
            # real similarity, and not ``weights``. ``r`` and ``c`` are positions in the
            # table, so ``left_a[r]`` and ``left_b[c]`` are the real unit numbers.
            # Example: sims[r, c] == 0.93 is kept, and 0.22 is thrown away.
            if single or sims[r, c] >= threshold:
                pairs[left_a[int(r)]] = left_b[int(c)]
                scores[left_a[int(r)]] = float(sims[r, c])  # float(): a plain Python number

    # Step 3: join the units of split and joined sentences.
    # Each pair becomes a cluster (unit numbers of text A, unit numbers of text B). The
    # numbers are in lists because a cluster can grow. Example: [([1], [0]), ([0], [1])].
    clusters = [([a], [b]) for a, b in pairs.items()]
    # _cluster adds a lonely unit to a neighboring cluster when that makes the match clearly
    # better. It changes ``clusters`` and returns the positions of the clusters that grew.
    merged = _cluster(units_a, units_b, clusters, embedder)
    # _assemble gives each cluster its status and adds the deleted and inserted units.
    return _assemble(units_a, units_b, clusters, scores, merged, embedder)


def _similarity(embedder: Embedder, a: Sequence[Unit], b: Sequence[Unit]) -> float:
    """The cosine similarity of the joined texts of two groups of units.

    The units of each group are joined with spaces and encoded as one text. This is used
    when a group has more than one unit, for example a sentence that text B split in two.

    Example: "Users can cancel anytime, with a full refund within 14 days." has the
    similarity 0.75 with the first half only ("Users can cancel anytime."), and 0.97 with
    both halves ("Users can cancel anytime. A full refund is paid within 14 days.").
    """
    va, vb = embedder.encode([_join(a), _join(b)])  # two vectors, one for each group
    return float(va @ vb)  # the vectors have the length 1, so this is the cosine similarity


def _cluster(
    units_a: list[Unit],
    units_b: list[Unit],
    clusters: list[tuple[list[int], list[int]]],
    embedder: Embedder,
) -> set[int]:
    """Add each unmatched unit to an adjacent cluster if the similarity increases.

    A cluster is a pair (indexes in text A, indexes in text B). The function changes
    ``clusters`` in place and returns the positions of the clusters that grew.

    Why it is needed: step 2 pairs one unit with one unit. If text B split one sentence of
    text A in two, the second half has no partner and would look like new text. Joining it
    to the first half shows that the two halves together are the old sentence.

    Example: text A has "Users can cancel anytime, with a full refund within 14 days." and
    text B has "Users can cancel anytime." and "A full refund is paid within 14 days.".
    Step 2 pairs A0 with B0 (similarity 0.75), and B1 is alone. With B1 added, the
    similarity is 0.97. The gain, about 0.23, is more than ``MERGE_GAIN``, so the cluster
    ``([0], [0])`` becomes ``([0], [0, 1])`` and the function returns ``{0}``.
    A unit that makes the match worse is not joined.
    """
    merged: set[int] = set()  # positions in ``clusters`` of the clusters that grew
    for side, units in enumerate((units_a, units_b)):  # side 0 is text A, side 1 is text B
        # The units of this text that are in a cluster already.
        used = {i for cluster in clusters for i in cluster[side]}
        for index in range(len(units)):
            if index in used:
                continue  # this unit has a partner already
            # This unit is alone. Look for a cluster with the unit just before or just after
            # it: only a neighbor can be the other half of a split sentence.
            for position, cluster in enumerate(clusters):
                members = cluster[side]  # the units of this text in the cluster
                if not any(abs(index - member) == 1 for member in members):
                    continue
                a = [units_a[k] for k in sorted(cluster[0])]  # the cluster in text A
                b = [units_b[k] for k in sorted(cluster[1])]  # the cluster in text B
                # The units of this side with the lonely unit added, in text order.
                grown = [units[k] for k in sorted([*members, index])]
                before = _similarity(embedder, a, b)
                # Only the side of the lonely unit grows. The other side stays the same.
                after = (
                    _similarity(embedder, grown, b)
                    if side == 0
                    else _similarity(embedder, a, grown)
                )
                # Join only if the match gets clearly better (more than MERGE_GAIN).
                if after - before > MERGE_GAIN:
                    members.append(index)  # the cluster grows (this changes ``clusters``)
                    used.add(index)
                    merged.add(position)  # remember it: its similarity is measured again later
                    break  # the unit has a place now, so do not try the other clusters
    return merged


def _assemble(
    units_a: list[Unit],
    units_b: list[Unit],
    clusters: list[tuple[list[int], list[int]]],
    scores: dict[int, float],
    merged: set[int],
    embedder: Embedder,
) -> list[AlignedUnit]:
    """Build the aligned units in the order of text B.

    Each cluster gets the status unchanged, moved or modified. An unmatched unit of text A
    becomes a deleted unit, and an unmatched unit of text B becomes an inserted unit.

    The steps:

    1. Sort the clusters by their place in text B, because the result follows text B.
    2. Find the clusters that kept their order. Read in the order of text B, the places in
       text A should go up (0, 1, 2). The longest run that goes up kept its order, and the
       other clusters moved.
    3. Give each cluster a status. The same words and not moved is ``unchanged``. The same
       words and moved is ``moved``. Different words is ``modified`` (and ``moved`` can still
       be True). The similarity is 1.0 for the same words, it is measured again for a
       cluster that grew, and otherwise it is the score from step 2.
    4. Add the units without a partner. A deleted unit has no place in text B, so it goes
       just before the match of the next unit of text A. An inserted unit goes at its own
       place in text B.
    5. Sort everything by its sort key and number it from 1.

    Example. Text A: refunds, shipping, support, EU data. Text B: shipping, refunds,
    support, cookies. The result:

        id 1  unchanged  shipping
        id 2  modified   refunds (moved: first in text A, second in text B)
        id 3  modified   support
        id 4  inserted   cookies
        id 5  deleted    EU data (the last unit of text A, so it goes to the end)
    """
    # Step 1: follow the order of text B. A cluster is placed by its first unit in text B.
    ordered = sorted(clusters, key=lambda c: min(c[1]))
    # Step 2: the places in text A, read in the order of text B. Example: [1, 0, 2]. The
    # longest run that goes up is {1, 2}, so those clusters kept their order. The cluster
    # of A0 moved.
    kept_order = _longest_increasing([min(c[0]) for c in ordered])
    # (sort key, status, original units, modified units, similarity, moved)
    placed: list[tuple[float, str, tuple[Unit, ...], tuple[Unit, ...], float, bool]] = []
    for cluster in ordered:
        a = tuple(units_a[k] for k in sorted(cluster[0]))  # the units of text A in the cluster
        b = tuple(units_b[k] for k in sorted(cluster[1]))  # the units of text B in the cluster
        # Step 3: do both sides have the same words? Case and punctuation do not count.
        same = [k for u in a for k in u.key] == [k for u in b for k in u.key]
        moved = min(cluster[0]) not in kept_order
        status = ("moved" if moved else "unchanged") if same else "modified"
        grew = clusters.index(cluster) in merged
        # A cluster that grew has a new joined text, so its similarity is measured again.
        score = 1.0 if same else _similarity(embedder, a, b) if grew else scores[cluster[0][0]]
        # The sort key of a cluster is its place in text B.
        placed.append((float(min(cluster[1])), status, a, b, round(score, 4), moved))

    # Step 4: the units without a partner.
    used_a = {i for c in clusters for i in c[0]}  # the units of text A that are in a cluster
    used_b = {j for c in clusters for j in c[1]}  # the units of text B that are in a cluster
    # (place in text A, place in text B) of every cluster, in the order of text A.
    anchors = sorted((min(c[0]), min(c[1])) for c in clusters)
    for i, unit in enumerate(units_a):
        if i not in used_a:  # a deleted unit goes before the match of the next unit of text A
            # The anchor is the place in text B of the next cluster after this unit in text
            # A. If there is none, the unit goes to the end.
            anchor = next((j for k, j in anchors if k > i), len(units_b))
            # The sort key is just before the anchor (minus 0.5). The small term with ``i``
            # only keeps several deleted units in their order.
            placed.append(
                (anchor - 0.5 + i / (10 * len(units_a)), "deleted", (unit,), (), 0.0, False)
            )
    for j, unit in enumerate(units_b):
        if j not in used_b:  # an inserted unit goes at its own place in text B
            placed.append((float(j), "inserted", (), (unit,), 0.0, False))

    # Step 5: sort by the sort key and number the aligned units from 1.
    placed.sort(key=lambda item: item[0])
    return [
        AlignedUnit(id=n, status=status, original=a, modified=b, similarity=score, moved=moved)  # type: ignore[arg-type]
        for n, (_, status, a, b, score, moved) in enumerate(placed, 1)
    ]


def _longest_increasing(values: list[int]) -> set[int]:
    """The values on one longest increasing subsequence: the units that kept their order.

    "Increasing" means that each value is bigger than the one before. The values do not have
    to be next to each other. If several runs are equally long, one of them is returned.

    Examples:

        [0, 1, 2]     →  {0, 1, 2}   nothing moved
        [1, 0, 2]     →  {1, 2}      the unit at place 0 moved
        [3, 0, 1, 2]  →  {0, 1, 2}   the unit at place 3 moved to the front
        [2, 1, 0]     →  {2}         the order is reversed: only one unit kept its place

    How it works: ``length[i]`` is the length of the longest increasing run that ends at
    ``values[i]``, and ``previous[i]`` is the index of the value before it in that run.
    The run is found by starting at the best end and following ``previous`` back.
    """
    if not values:
        return set()
    length = [1] * len(values)  # every value alone is a run of length 1
    previous = [-1] * len(values)  # -1 means that the run starts at this value
    for i in range(len(values)):
        for j in range(i):  # every value before ``i``
            # values[i] can continue the run that ends at values[j] if it is bigger, and if
            # that makes the run longer than the best run found for ``i`` so far.
            if values[j] < values[i] and length[j] + 1 > length[i]:
                length[i], previous[i] = length[j] + 1, j
    k = max(range(len(values)), key=length.__getitem__)  # the index where a longest run ends
    keep: set[int] = set()
    while k >= 0:  # follow ``previous`` back to the start of the run
        keep.add(values[k])
        k = previous[k]
    return keep


def _join(units: Sequence[Unit]) -> str:
    """The text of a group of units, joined with a space.

    Example: the units "Users can cancel anytime." and "Fees apply." give
    "Users can cancel anytime. Fees apply.".
    """
    return " ".join(u.text for u in units)


def _tokens(units: Sequence[Unit]) -> list[tuple[str, bool, bool]]:
    """(text, followed by a space, is punctuation) for each token of the units.

    The word edits are found on these tokens. The spacing lets us write an edit as the user
    typed it. The last token of a unit counts as followed by a space, so the words of two
    units do not run together.

    Example: the unit "The fee is $5." gives

        [("The", True, False), ("fee", True, False), ("is", True, False),
         ("$", False, False), ("5", False, False), (".", True, True)]

    There is no space between "$" and "5", and "." is punctuation.
    """
    out: list[tuple[str, bool, bool]] = []
    for unit in units:
        out.extend(zip(unit.tokens, unit.spaces, unit.punct, strict=True))
        if out:
            out[-1] = (out[-1][0], True, out[-1][2])  # the end of a unit counts as a space
    return out


def _whole_words(
    a: Sequence[tuple[str, bool, bool]],
    b: Sequence[tuple[str, bool, bool]],
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> tuple[int, int, int, int]:
    """Widen an edit to whole words: "added 'non-'" becomes "'refundable' → 'non-refundable'".

    The edit grows only over tokens that are the same on both sides, are not
    punctuation, and attach to the edit without a space.

    Example: "The fee is $5." changes to "The fee is $10.". The matcher sees only "5" → "10",
    because the "$" is the same on both sides. The "$" has no space before the "5", so the
    edit grows to "$5" → "$10".

    Args:
        a: The tokens of the original text, from ``_tokens``.
        b: The tokens of the modified text, from ``_tokens``.
        i0: The start of the edit in ``a``.
        i1: The end of the edit in ``a``. The edit is ``a[i0:i1]``.
        j0: The start of the edit in ``b``.
        j1: The end of the edit in ``b``. The edit is ``b[j0:j1]``.

    Returns:
        The new ``i0, i1, j0, j1``, each as far as the edit has grown.
    """

    def attached_left(tokens: Sequence[tuple[str, bool, bool]], k: int) -> bool:
        """True if token ``k`` follows the previous token without a space."""
        return k > 0 and not tokens[k - 1][1]

    # Grow to the left while the token before the edit is the same on both sides, is not
    # punctuation, and is attached (without a space) to the edit in ``a`` or in ``b``.
    while (
        i0 > 0 and j0 > 0 and a[i0 - 1][0].lower() == b[j0 - 1][0].lower()
        and not a[i0 - 1][2] and (attached_left(a, i0) or attached_left(b, j0))
    ):  # fmt: skip
        i0, j0 = i0 - 1, j0 - 1
    # Grow to the right in the same way: the token after the edit must be the same on both
    # sides, must not be punctuation, and must follow the edit without a space.
    while (
        i1 < len(a) and j1 < len(b) and a[i1][0].lower() == b[j1][0].lower()
        and not a[i1][2] and ((i1 > i0 and not a[i1 - 1][1]) or (j1 > j0 and not b[j1 - 1][1]))
    ):  # fmt: skip
        i1, j1 = i1 + 1, j1 + 1
    return i0, i1, j0, j1


def _render(tokens: Sequence[tuple[str, bool, bool]]) -> str:
    """The tokens with their original spacing, for example "$5" or "non-refundable".

    A token is followed by a space only if there was a space in the text. ``strip`` removes
    the space after the last token.

    Example: the tokens "is", "$", "5" and "and" give "is $5 and".
    """
    return "".join(text + (" " if space else "") for text, space, _ in tokens).strip()
