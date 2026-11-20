"""Scoring shared by every model, so the comparison is like for like."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score

INVALID = "<invalid>"


def score(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    """Accuracy, macro-F1 (each component counts equally) and the share of unusable answers.

    Predictions outside the label set count as wrong; ``invalid_rate`` reports how often a
    generative model answered with something that is not a component name.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    labels = sorted(set(gold))
    return {
        "accuracy": round(float(accuracy_score(gold, predicted)), 4),
        "macro_f1": round(
            float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)), 4
        ),
        "invalid_rate": round(sum(p == INVALID for p in predicted) / len(predicted), 4),
        "n": len(gold),
    }
