"""Stage 5 - structured output: a Markdown report for human reviewers."""

from __future__ import annotations

from semantic_change.models import ChangeResult, UnitView

# Status labels in the report.
_STATUS = {
    "modified": "Modified",
    "inserted": "Added",
    "deleted": "Removed",
    "moved": "Moved",
}


def render_markdown(result: ChangeResult, units: list[UnitView]) -> str:
    """Render the verdict and the changed units as a Markdown report.

    The report has a verdict table, the summary and the evidence, one section for each
    change in meaning, and lists of the reworded units and the moved units.
    """
    lines = [
        "# Semantic change report",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Meaning changed | {'**Yes**' if result.change_detected else 'No'} |",
        f"| Change type | `{result.change_type.value}` |",
        f"| Severity | {_severity(result.severity.value)} |",
        f"| Confidence | {result.confidence:.2f} |",
        "",
        f"**Summary.** {_flat(result.summary)}",
        "",
    ]
    if result.evidence:
        lines += ["**Evidence**", "", *[f"- `{_code(e)}`" for e in result.evidence], ""]

    changed = [u for u in units if u.finding is not None and u.finding.change_detected]
    reworded = [u for u in units if u.finding is not None and not u.finding.change_detected]
    moved = [u for u in units if u.status == "moved"]
    lines += [f"## Changes ({len(changed)})", ""]
    if not changed:
        lines += ["No unit changed its meaning.", ""]
    for number, unit in enumerate(changed, 1):
        lines += _change(number, unit)
    if reworded:
        lines += [f"## Reworded, same meaning ({len(reworded)})", ""]
        lines += [f"- {_quote(u.original)} → {_quote(u.modified)}" for u in reworded]
        lines.append("")
    if moved:
        lines += [f"## Moved without a change ({len(moved)})", ""]
        lines += [f"- {_quote(u.modified)}" for u in moved]
        lines.append("")
    return "\n".join(lines)


def _change(number: int, unit: UnitView) -> list[str]:
    """The report section of one unit whose meaning changed."""
    finding = unit.finding
    assert finding is not None
    heading = finding.change_type.value.replace("_", " ").capitalize()
    lines = [
        f"### {number}. {heading} — {_severity(finding.severity.value)} "
        f"(confidence {finding.confidence:.2f})",
        "",
        f"- **Unit:** {_STATUS[unit.status]}, similarity {unit.similarity:.2f}"
        + (", moved" if unit.moved else ""),
    ]
    if unit.original:
        lines.append(f"- **Original:** {_quote(unit.original)}")
    if unit.modified:
        lines.append(f"- **Modified:** {_quote(unit.modified)}")
    lines.append(f"- **What changed:** {_flat(finding.summary)}")
    if finding.risk:
        lines.append(f"- **Why it matters:** {_flat(finding.risk)}")
    if finding.evidence:
        lines.append("- **Evidence:** " + ", ".join(f"`{_code(e)}`" for e in finding.evidence))
    return [*lines, ""]


def _severity(value: str) -> str:
    """The severity label. A high severity is bold."""
    return f"**{value.upper()}**" if value == "high" else value.capitalize()


def _quote(text: str) -> str:
    """The text on one line, in quotation marks."""
    return "“" + _flat(text) + "”"


def _flat(text: str) -> str:
    """The text on one line, with "|" escaped for Markdown tables."""
    return " ".join(text.split()).replace("|", "\\|")


def _code(text: str) -> str:
    """The text on one line for inline code. A backtick becomes an apostrophe."""
    return " ".join(text.split()).replace("`", "'")
