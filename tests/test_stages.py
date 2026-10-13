"""Stages 1-3 with the real models: segmentation, embedding, alignment and clustering."""

from __future__ import annotations

from semantic_change.alignment import align
from semantic_change.embeddings import SentenceEmbedder
from semantic_change.segmentation import ClauseSegmenter


def texts(segmenter: ClauseSegmenter, value: str) -> list[str]:
    return [u.text for u in segmenter.segment(value)]


# ------------------------------------------------------------- segmentation
def test_coordinated_clause_with_own_subject_is_a_unit(segmenter: ClauseSegmenter) -> None:
    assert texts(segmenter, "Users can cancel their plan, but the setup fee is not refunded.") == [
        "Users can cancel their plan,",
        "but the setup fee is not refunded.",
    ]


def test_subordinate_clause_stays_with_its_clause(segmenter: ClauseSegmenter) -> None:
    sentence = "Members can book the room if they have manager approval."
    assert texts(segmenter, sentence) == [sentence]


def test_shared_subject_is_not_split(segmenter: ClauseSegmenter) -> None:
    sentence = "We may update this policy and will notify you."
    assert texts(segmenter, sentence) == [sentence]


def test_line_breaks_end_units(segmenter: ClauseSegmenter) -> None:
    assert texts(segmenter, "- Refunds take 5 days\n- Shipping is free") == [
        "- Refunds take 5 days",
        "- Shipping is free",
    ]


def test_empty_text_has_no_units(segmenter: ClauseSegmenter) -> None:
    assert segmenter.segment("  \n ") == []


# ------------------------------------------------------- embedding & alignment
def test_embeddings_are_normalized(embedder: SentenceEmbedder) -> None:
    vectors = embedder.encode(["The shop sells used books.", "The shop sells used books."])
    assert abs(float(vectors[0] @ vectors[0]) - 1.0) < 1e-4
    paraphrase, unrelated = embedder.encode(
        ["The store sells second-hand books.", "The museum opens at 9 AM."]
    )
    assert float(vectors[0] @ paraphrase) > 0.6 > float(vectors[0] @ unrelated)


def test_alignment_finds_moved_modified_inserted_and_deleted_units(
    segmenter: ClauseSegmenter, embedder: SentenceEmbedder
) -> None:
    a = (
        "Refunds are available within 30 days. Shipping is free for all orders. "
        "Support answers questions by email. Data is stored in the EU."
    )
    b = (
        "Shipping is free for all orders. Refunds are available within 14 days. "
        "Support answers questions by email and chat. Cookies track page visits."
    )
    units = align(segmenter.segment(a), segmenter.segment(b), embedder)
    by_status = {(u.status, u.original_text, u.modified_text) for u in units}
    assert ("modified", "Refunds are available within 30 days.",
            "Refunds are available within 14 days.") in by_status  # fmt: skip
    assert ("deleted", "Data is stored in the EU.", "") in by_status
    assert ("inserted", "", "Cookies track page visits.") in by_status
    refund = next(u for u in units if u.original_text.startswith("Refunds"))
    assert refund.word_edits() == ["'30' → '14'"]
    assert [u.id for u in units] == list(range(1, len(units) + 1))


def test_identical_sentences_in_a_new_order_are_moved(
    segmenter: ClauseSegmenter, embedder: SentenceEmbedder
) -> None:
    units = align(
        segmenter.segment("Orders ship daily. Returns are free. Support is open daily."),
        segmenter.segment("Returns are free. Orders ship daily. Support is open daily."),
        embedder,
    )
    assert sorted(u.status for u in units) == ["moved", "unchanged", "unchanged"]
    assert not any(u.flagged for u in units)


def test_split_sentence_joins_one_cluster(
    segmenter: ClauseSegmenter, embedder: SentenceEmbedder
) -> None:
    units = align(
        segmenter.segment("Users can cancel anytime, with a full refund within 14 days."),
        segmenter.segment("Users can cancel anytime. A full refund is paid within 14 days."),
        embedder,
    )
    assert len(units) == 1
    assert units[0].modified_text == (
        "Users can cancel anytime. A full refund is paid within 14 days."
    )


def test_word_edits_keep_the_original_spelling(
    segmenter: ClauseSegmenter, embedder: SentenceEmbedder
) -> None:
    fee, deposit = align(
        segmenter.segment("The fee is $5 and the deposit is refundable."),
        segmenter.segment("The fee is $10 and the deposit is non-refundable."),
        embedder,
    )
    assert fee.word_edits() == ["'$5' → '$10'"]
    assert deposit.word_edits() == ["'refundable' → 'non-refundable'"]


def test_two_one_clause_texts_always_pair(
    segmenter: ClauseSegmenter, embedder: SentenceEmbedder
) -> None:
    [unit] = align(
        segmenter.segment("The museum opens at 9 AM."),
        segmenter.segment("Photography is not permitted."),
        embedder,
    )
    assert unit.status == "modified"
    assert unit.similarity < 0.45
