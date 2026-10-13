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

# Weight of the position penalty. Positions are scaled to the range 0-1.
POSITION_WEIGHT = 0.05
# Minimum increase of similarity for an unmatched unit to join a neighboring match.
MERGE_GAIN = 0.05


class Embedder(Protocol):
    """Encodes texts to normalized vectors, for example ``SentenceEmbedder``."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class AlignedUnit:
    """A group of corresponding units, usually one unit of text A and one of text B.

    An inserted unit has no original units. A deleted unit has no modified units.
    """

    id: int  # position in the alignment, from 1
    status: UnitStatus
    original: tuple[Unit, ...]  # units of text A
    modified: tuple[Unit, ...]  # units of text B
    similarity: float  # cosine similarity of the two texts; 1.0 for the same wording
    moved: bool = False  # the unit is not in the same order as in text A

    @property
    def original_text(self) -> str:
        """The text of the original units."""
        return " ".join(u.text for u in self.original)

    @property
    def modified_text(self) -> str:
        """The text of the modified units."""
        return " ".join(u.text for u in self.modified)

    @property
    def flagged(self) -> bool:
        """True if the wording changed. The evaluation agent must judge each flagged unit."""
        return self.status in ("modified", "inserted", "deleted")

    def word_edits(self) -> list[str]:
        """Word-level edits from the original text to the modified text.

        Examples: "'anytime' → 'within 7 days'", "added 'not'" and "removed 'only'". An
        inserted or deleted unit gives one "added: ..." or "removed: ..." item.
        """
        if not self.original:
            return [f"added: {self.modified_text}"]
        if not self.modified:
            return [f"removed: {self.original_text}"]
        a, b = _tokens(self.original), _tokens(self.modified)
        matcher = SequenceMatcher(
            a=[t.lower() for t, _, _ in a], b=[t.lower() for t, _, _ in b], autojunk=False
        )
        edits = []
        for tag, i0, i1, j0, j1 in matcher.get_opcodes():
            if tag == "equal" or all(p for _, _, p in (*a[i0:i1], *b[j0:j1])):
                continue  # a punctuation-only edit does not change the meaning
            i0, i1, j0, j1 = _whole_words(a, b, i0, i1, j0, j1)
            old, new = _render(a[i0:i1]), _render(b[j0:j1])
            if old and new:
                edits.append(f"'{old}' → '{new}'")
            elif old:
                edits.append(f"removed '{old}'")
            else:
                edits.append(f"added '{new}'")
        return edits


def align(
    units_a: list[Unit],
    units_b: list[Unit],
    embedder: Embedder,
    threshold: float = 0.45,
) -> list[AlignedUnit]:
    """Match the units of text A to the units of text B.

    Args:
        units_a: The units of the original text (stage 1).
        units_b: The units of the modified text (stage 1).
        embedder: The encoder of stage 2.
        threshold: The minimum cosine similarity of a match.

    Returns:
        The aligned units in the order of text B. A deleted unit comes before the match
        of the next unit of text A.
    """
    # Step 1: units with the same wording.
    pairs: dict[int, int] = {}
    scores: dict[int, float] = {}
    keys = SequenceMatcher(a=[u.key for u in units_a], b=[u.key for u in units_b], autojunk=False)
    for block in keys.get_matching_blocks():
        for k in range(block.size):
            pairs[block.a + k] = block.b + k
            scores[block.a + k] = 1.0

    # Step 2: Hungarian matching of the other units.
    left_a = [i for i in range(len(units_a)) if i not in pairs]
    used_b = set(pairs.values())
    left_b = [j for j in range(len(units_b)) if j not in used_b]
    if left_a and left_b:
        va = embedder.encode([units_a[i].text for i in left_a])
        vb = embedder.encode([units_b[j].text for j in left_b])
        sims = va @ vb.T
        pos_a = np.array([i / max(len(units_a) - 1, 1) for i in left_a])
        pos_b = np.array([j / max(len(units_b) - 1, 1) for j in left_b])
        weights = sims - POSITION_WEIGHT * np.abs(pos_a[:, None] - pos_b[None, :])
        rows, cols = linear_sum_assignment(weights, maximize=True)
        single = len(units_a) == len(units_b) == 1  # two one-unit texts always correspond
        for r, c in zip(rows, cols, strict=True):
            if single or sims[r, c] >= threshold:
                pairs[left_a[int(r)]] = left_b[int(c)]
                scores[left_a[int(r)]] = float(sims[r, c])

    # Step 3: clustering of split and joined sentences.
    clusters = [([a], [b]) for a, b in pairs.items()]
    merged = _cluster(units_a, units_b, clusters, embedder)
    return _assemble(units_a, units_b, clusters, scores, merged, embedder)


def _similarity(embedder: Embedder, a: Sequence[Unit], b: Sequence[Unit]) -> float:
    """The cosine similarity of the joined texts of two groups of units."""
    va, vb = embedder.encode([_join(a), _join(b)])
    return float(va @ vb)


def _cluster(
    units_a: list[Unit],
    units_b: list[Unit],
    clusters: list[tuple[list[int], list[int]]],
    embedder: Embedder,
) -> set[int]:
    """Add each unmatched unit to an adjacent cluster if the similarity increases.

    A cluster is a pair (indexes in text A, indexes in text B). The function changes
    ``clusters`` in place and returns the positions of the clusters that grew.
    """
    merged: set[int] = set()
    for side, units in enumerate((units_a, units_b)):
        used = {i for cluster in clusters for i in cluster[side]}
        for index in range(len(units)):
            if index in used:
                continue
            for position, cluster in enumerate(clusters):
                members = cluster[side]
                if not any(abs(index - member) == 1 for member in members):
                    continue
                a = [units_a[k] for k in sorted(cluster[0])]
                b = [units_b[k] for k in sorted(cluster[1])]
                grown = [units[k] for k in sorted([*members, index])]
                before = _similarity(embedder, a, b)
                after = (
                    _similarity(embedder, grown, b)
                    if side == 0
                    else _similarity(embedder, a, grown)
                )
                if after - before > MERGE_GAIN:
                    members.append(index)
                    used.add(index)
                    merged.add(position)
                    break
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
    """
    ordered = sorted(clusters, key=lambda c: min(c[1]))
    # Clusters on the longest run in text A order kept their order. The others moved.
    kept_order = _longest_increasing([min(c[0]) for c in ordered])
    # (sort key, status, original units, modified units, similarity, moved)
    placed: list[tuple[float, str, tuple[Unit, ...], tuple[Unit, ...], float, bool]] = []
    for cluster in ordered:
        a = tuple(units_a[k] for k in sorted(cluster[0]))
        b = tuple(units_b[k] for k in sorted(cluster[1]))
        same = [k for u in a for k in u.key] == [k for u in b for k in u.key]
        moved = min(cluster[0]) not in kept_order
        status = ("moved" if moved else "unchanged") if same else "modified"
        grew = clusters.index(cluster) in merged
        # A cluster that grew has a new joined text, so its similarity is measured again.
        score = 1.0 if same else _similarity(embedder, a, b) if grew else scores[cluster[0][0]]
        placed.append((float(min(cluster[1])), status, a, b, round(score, 4), moved))

    used_a = {i for c in clusters for i in c[0]}
    used_b = {j for c in clusters for j in c[1]}
    anchors = sorted((min(c[0]), min(c[1])) for c in clusters)
    for i, unit in enumerate(units_a):
        if i not in used_a:  # a deleted unit goes before the match of the next unit of text A
            anchor = next((j for k, j in anchors if k > i), len(units_b))
            placed.append(
                (anchor - 0.5 + i / (10 * len(units_a)), "deleted", (unit,), (), 0.0, False)
            )
    for j, unit in enumerate(units_b):
        if j not in used_b:
            placed.append((float(j), "inserted", (), (unit,), 0.0, False))

    placed.sort(key=lambda item: item[0])
    return [
        AlignedUnit(id=n, status=status, original=a, modified=b, similarity=score, moved=moved)  # type: ignore[arg-type]
        for n, (_, status, a, b, score, moved) in enumerate(placed, 1)
    ]


def _longest_increasing(values: list[int]) -> set[int]:
    """The values on one longest increasing subsequence: the units that kept their order."""
    if not values:
        return set()
    length = [1] * len(values)
    previous = [-1] * len(values)
    for i in range(len(values)):
        for j in range(i):
            if values[j] < values[i] and length[j] + 1 > length[i]:
                length[i], previous[i] = length[j] + 1, j
    k = max(range(len(values)), key=length.__getitem__)
    keep: set[int] = set()
    while k >= 0:
        keep.add(values[k])
        k = previous[k]
    return keep


def _join(units: Sequence[Unit]) -> str:
    """The text of a group of units."""
    return " ".join(u.text for u in units)


def _tokens(units: Sequence[Unit]) -> list[tuple[str, bool, bool]]:
    """(text, followed by a space, is punctuation) for each token of the units."""
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
    """

    def attached_left(tokens: Sequence[tuple[str, bool, bool]], k: int) -> bool:
        """True if token ``k`` follows the previous token without a space."""
        return k > 0 and not tokens[k - 1][1]

    while (
        i0 > 0 and j0 > 0 and a[i0 - 1][0].lower() == b[j0 - 1][0].lower()
        and not a[i0 - 1][2] and (attached_left(a, i0) or attached_left(b, j0))
    ):  # fmt: skip
        i0, j0 = i0 - 1, j0 - 1
    while (
        i1 < len(a) and j1 < len(b) and a[i1][0].lower() == b[j1][0].lower()
        and not a[i1][2] and ((i1 > i0 and not a[i1 - 1][1]) or (j1 > j0 and not b[j1 - 1][1]))
    ):  # fmt: skip
        i1, j1 = i1 + 1, j1 + 1
    return i0, i1, j0, j1


def _render(tokens: Sequence[tuple[str, bool, bool]]) -> str:
    """The tokens with their original spacing, for example "$5" or "non-refundable"."""
    return "".join(text + (" " if space else "") for text, space, _ in tokens).strip()
