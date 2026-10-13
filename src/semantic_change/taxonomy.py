"""Change types and severity levels.

The tool schemas of the agent, the system prompt and the API responses all use these
values, so this module is the only place that defines them.
"""

from __future__ import annotations

from enum import StrEnum


class ChangeType(StrEnum):
    """Category of a change between two texts."""

    NO_CHANGE = "no_change"
    PARAPHRASE = "paraphrase"
    MEANING_REVERSED = "meaning_reversed"
    RESTRICTION_ADDED = "restriction_added"
    RESTRICTION_REMOVED = "restriction_removed"
    OBLIGATION_STRENGTHENED = "obligation_strengthened"
    OBLIGATION_WEAKENED = "obligation_weakened"
    NUMERIC_CHANGE = "numeric_change"
    ENTITY_CHANGE = "entity_change"
    CONTENT_CHANGE = "content_change"
    INFORMATION_ADDED = "information_added"
    INFORMATION_REMOVED = "information_removed"

    @property
    def is_meaningful(self) -> bool:
        """True if the category is a change in meaning (``change_detected``)."""
        return self not in (ChangeType.NO_CHANGE, ChangeType.PARAPHRASE)


class Severity(StrEnum):
    """Impact of a change on a reader who relies on the text."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Definition and example pair of each category. The system prompt contains them.
DEFINITIONS: dict[ChangeType, tuple[str, str, str]] = {
    # category: (definition, example from text A, example from text B)
    ChangeType.NO_CHANGE: (
        "Identical apart from case, spacing or punctuation.",
        "Users can cancel anytime.",
        "users can cancel anytime",
    ),
    ChangeType.PARAPHRASE: (
        "Different wording, same meaning (synonyms, voice, order, equivalent values).",
        "Users must log in to continue.",
        "Login is required to continue.",
    ),
    ChangeType.MEANING_REVERSED: (
        "The statement now says the opposite: negation, antonym, or permission "
        "became prohibition (or the reverse).",
        "Users may share their credentials.",
        "Users may not share their credentials.",
    ),
    ChangeType.RESTRICTION_ADDED: (
        "A new limit narrows what is allowed or available: time window, condition, "
        "exception, 'only', cap, approval step, fee or narrower scope.",
        "Users can cancel their subscription anytime.",
        "Users can cancel their subscription within 7 days.",
    ),
    ChangeType.RESTRICTION_REMOVED: (
        "A limit was removed or loosened, or a prohibition or denial got an exception.",
        "Refunds are available within 14 days of purchase.",
        "Refunds are available at any time.",
    ),
    ChangeType.OBLIGATION_STRENGTHENED: (
        "A duty became stronger or new (optional or recommended became mandatory).",
        "Employees may attend the training.",
        "Employees must attend the training.",
    ),
    ChangeType.OBLIGATION_WEAKENED: (
        "A duty or commitment became weaker or applies in fewer cases ('will' -> 'may').",
        "We will notify you of a breach.",
        "We may notify you of a breach.",
    ),
    ChangeType.NUMERIC_CHANGE: (
        "A quantity changed: amount, percentage, count, duration, rate, date or time.",
        "The monthly fee is $10.",
        "The monthly fee is $15.",
    ),
    ChangeType.ENTITY_CHANGE: (
        "A party, object, product or place was replaced, or two parties swapped roles.",
        "Only administrators can approve refunds.",
        "Only managers can approve refunds.",
    ),
    ChangeType.CONTENT_CHANGE: (
        "The meaning changed in another way, for example a different action or an "
        "unrelated statement.",
        "Users can view the reports.",
        "Users can edit the reports.",
    ),
    ChangeType.INFORMATION_ADDED: (
        "New detail, list item or clause without a specific restriction or duty effect.",
        "The app is available on iOS.",
        "The app is available on iOS and Android.",
    ),
    ChangeType.INFORMATION_REMOVED: (
        "Detail, a list item or a clause was removed.",
        "The package includes a charger and a case.",
        "The package includes a charger.",
    ),
}

# Severity rules for the system prompt.
SEVERITY_RUBRIC: dict[Severity, str] = {
    Severity.HIGH: "Changes rights, duties, permissions, money or deadlines, or reverses the "
    "meaning. A reader who misses it can act wrongly or lose money.",
    Severity.MEDIUM: "Changes facts, parties, scope or details with moderate impact.",
    Severity.LOW: "No or minor effect on meaning: rewording or small clarifications.",
}
