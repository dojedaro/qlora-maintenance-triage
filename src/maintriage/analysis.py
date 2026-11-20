"""How much to trust the headline numbers: uncertainty, significance and leakage checks.

    python -m maintriage.analysis --results results

Reads ``test_predictions.csv`` from a finished run and writes ``analysis.json``:

* 95% bootstrap confidence intervals for accuracy and macro-F1 of every model;
* McNemar's exact test between the fine-tuned LLM and the TF-IDF baseline (paired, same reports);
* strict accuracy: the raw answer must be exactly a component name, no lenient parsing;
* a leakage check: accuracy on reports that do not mention their own component's name, plus a
  keyword baseline that predicts whichever component name the report mentions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

MODELS = ("tfidf_logreg", "zero_shot", "qlora")


def bootstrap_ci(
    gold: np.ndarray, predicted: np.ndarray, rounds: int = 2000, seed: int = 0
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    labels = np.unique(gold)
    accuracy, macro_f1 = [], []
    for _ in range(rounds):
        idx = rng.integers(0, len(gold), len(gold))
        accuracy.append(np.mean(gold[idx] == predicted[idx]))
        macro_f1.append(
            f1_score(gold[idx], predicted[idx], labels=labels, average="macro", zero_division=0)
        )
    return {
        "accuracy_95ci": [round(float(np.percentile(accuracy, q)), 4) for q in (2.5, 97.5)],
        "macro_f1_95ci": [round(float(np.percentile(macro_f1, q)), 4) for q in (2.5, 97.5)],
    }


def paired_macro_f1_difference(
    gold: np.ndarray, a: np.ndarray, b: np.ndarray, rounds: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Macro-F1 of ``a`` minus ``b``, with a 95% interval from resampling the same reports for
    both models (paired), which is tighter and more honest than comparing two separate intervals."""
    rng = np.random.default_rng(seed)
    labels = np.unique(gold)

    def macro(g: np.ndarray, p: np.ndarray) -> float:
        return f1_score(g, p, labels=labels, average="macro", zero_division=0)

    diffs = []
    for _ in range(rounds):
        idx = rng.integers(0, len(gold), len(gold))
        diffs.append(macro(gold[idx], a[idx]) - macro(gold[idx], b[idx]))
    return {
        "difference": round(float(macro(gold, a) - macro(gold, b)), 4),
        "ci95": [round(float(np.percentile(diffs, q)), 4) for q in (2.5, 97.5)],
    }


def mcnemar(gold: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on the reports where exactly one of the two models is right."""
    only_a = int(np.sum((a == gold) & (b != gold)))
    only_b = int(np.sum((a != gold) & (b == gold)))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "p_value": round(float(p), 4)}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def mentions_label(fault: str, label: str) -> bool:
    """True if every word of the component name appears in the report."""
    return _words(label) <= _words(fault)


def keyword_baseline(faults: pd.Series, labels: list[str], fallback: str) -> np.ndarray:
    """Predict the longest component name fully mentioned in the report, else ``fallback``."""

    def pick(fault: str) -> str:
        hits = [label for label in labels if mentions_label(fault, label)]
        return max(hits, key=len) if hits else fallback

    return faults.map(pick).to_numpy()


def near_duplicate_check(predictions: pd.DataFrame, train_faults: pd.Series) -> dict:
    """Accuracy on test reports whose first six words (numbers ignored) match no training report.

    Templated reports ("TC reported TIMS showed ...") share openings legitimately; if accuracy
    held up only on the shared ones, the models would be matching templates, not reading faults.
    """
    from maintriage.data import duplicate_key

    def prefix(text: str) -> str:
        return " ".join(duplicate_key(text).split()[:6])

    seen = set(train_faults.map(prefix))
    novel = ~predictions["fault"].map(prefix).isin(seen).to_numpy()
    gold = predictions["gold"].to_numpy()
    return {
        "n_novel_opening": int(novel.sum()),
        "accuracy_on_novel_opening": {
            m: round(float(np.mean(predictions[m].to_numpy()[novel] == gold[novel])), 4)
            for m in MODELS
            if m in predictions
        },
    }


def analyze(predictions: pd.DataFrame) -> dict:
    gold = predictions["gold"].to_numpy()
    labels = sorted(set(gold))
    models = [m for m in MODELS if m in predictions]
    result: dict = {"n": len(gold), "bootstrap": {}}
    for model in models:
        result["bootstrap"][model] = bootstrap_ci(gold, predictions[model].to_numpy())

    if "qlora" in predictions:
        result["mcnemar_qlora_vs_tfidf"] = mcnemar(
            gold, predictions["qlora"].to_numpy(), predictions["tfidf_logreg"].to_numpy()
        )
        result["macro_f1_qlora_minus_tfidf"] = paired_macro_f1_difference(
            gold, predictions["qlora"].to_numpy(), predictions["tfidf_logreg"].to_numpy()
        )
        result["qlora_tfidf_answers_differ"] = int(
            (predictions["qlora"] != predictions["tfidf_logreg"]).sum()
        )
    for model in ("zero_shot", "qlora"):
        raw = f"{model}_raw"
        if raw in predictions:
            exact = predictions[raw].fillna("").str.strip() == predictions["gold"]
            result[f"{model}_strict_accuracy"] = round(float(exact.mean()), 4)

    mentioned = np.array(
        [mentions_label(f, g) for f, g in zip(predictions["fault"], gold, strict=True)]
    )
    majority = pd.Series(gold).mode()[0]
    keyword = keyword_baseline(predictions["fault"], labels, majority)
    result["leakage"] = {
        "share_mentioning_own_component": round(float(mentioned.mean()), 4),
        "keyword_baseline_accuracy": round(float(np.mean(keyword == gold)), 4),
        "accuracy_on_reports_not_naming_component": {
            model: round(
                float(np.mean(predictions[model].to_numpy()[~mentioned] == gold[~mentioned])), 4
            )
            for model in models
        },
        "n_not_naming_component": int((~mentioned).sum()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--skip-near-duplicates", action="store_true", help="skip the check that downloads the data"
    )
    args = parser.parse_args()
    predictions = pd.read_csv(args.results / "test_predictions.csv")
    result = analyze(predictions)
    if not args.skip_near_duplicates:
        from maintriage.data import load_splits

        splits = load_splits()
        if splits.test["fault"].tolist() != predictions["fault"].tolist():
            raise SystemExit("the current split does not match the predictions file")
        result["near_duplicates"] = near_duplicate_check(predictions, splits.train["fault"])
    (args.results / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
