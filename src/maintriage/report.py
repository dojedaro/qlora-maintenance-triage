"""Charts for the README from a finished run's ``metrics.json`` and ``test_predictions.csv``.

python -m maintriage.report --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

NAMES = {
    "majority_class": "Majority class",
    "zero_shot_qwen": "Qwen2.5-1.5B zero-shot",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "qlora_qwen": "Qwen2.5-1.5B + QLoRA",
}
COLORS = {"qlora_qwen": "#2563eb"}
GREY = "#9ca3af"


def model_comparison(metrics: dict, path: Path) -> None:
    models = [m for m in NAMES if m in metrics["models"]]
    scores = metrics["models"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, key, title in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1"), strict=True
    ):
        values = [scores[m][key] for m in models]
        bars = ax.barh(
            [NAMES[m] for m in models], values, color=[COLORS.get(m, GREY) for m in models]
        )
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_visible(False)
    n = metrics["dataset"]["test"]
    k = metrics["dataset"]["components"]
    fig.suptitle(f"Held-out test set: {n} fault reports, {k} components", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def per_component(predictions: pd.DataFrame, path: Path) -> None:
    """Per-component F1 of the fine-tuned model against the classical baseline."""
    labels = predictions["gold"].value_counts().index.tolist()
    rows = []
    for model in ("qlora", "tfidf_logreg"):
        scores = f1_score(
            predictions["gold"], predictions[model], labels=labels, average=None, zero_division=0
        )
        rows.append(scores)
    frame = pd.DataFrame(rows, index=["QLoRA", "TF-IDF"], columns=labels).T
    frame = frame.sort_values("QLoRA")
    support = predictions["gold"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 7))
    y = range(len(frame))
    ax.barh([i + 0.2 for i in y], frame["QLoRA"], height=0.4, color="#2563eb", label="QLoRA")
    ax.barh([i - 0.2 for i in y], frame["TF-IDF"], height=0.4, color=GREY, label="TF-IDF")
    ax.set_yticks(list(y), [f"{c} ({support[c]})" for c in frame.index], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on the test set (test examples in brackets)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text())
    model_comparison(metrics, args.results / "model_comparison.png")
    predictions = pd.read_csv(args.results / "test_predictions.csv")
    if "qlora" in predictions:
        per_component(predictions, args.results / "per_component_f1.png")


if __name__ == "__main__":
    main()
