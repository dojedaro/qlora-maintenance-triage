"""One prompt format for zero-shot and fine-tuned runs, plus a strict answer parser."""

from __future__ import annotations

from collections.abc import Sequence

from maintriage.metrics import INVALID

SYSTEM = (
    "You triage train maintenance faults. Given a fault report, answer with the single "
    "component most likely to need repair, copied exactly from the list. Answer with the "
    "component name only."
)


def messages(fault: str, labels: Sequence[str]) -> list[dict[str, str]]:
    """Chat messages for one fault report. The label list is always included, so the base model
    sees the same choices the fine-tuned model was trained on."""
    options = "\n".join(f"- {label}" for label in labels)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Components:\n{options}\n\nFault report: {fault}"},
    ]


def parse_answer(text: str, labels: Sequence[str]) -> str:
    """Map a model's raw answer to a label, or ``INVALID``.

    Exact (case-insensitive) match on the first line first; otherwise the longest label that the
    answer contains, so "Saloon CCU Fan" is never read as "Saloon CCU". Anything else is invalid:
    no fuzzy matching that could hide a wrong or made-up answer.
    """
    first_line = text.strip().splitlines()[0].strip(" .\"'`*-") if text.strip() else ""
    by_lower = {label.lower(): label for label in labels}
    if first_line.lower() in by_lower:
        return by_lower[first_line.lower()]
    contained = [label for label in labels if label.lower() in first_line.lower()]
    return max(contained, key=len) if contained else INVALID
