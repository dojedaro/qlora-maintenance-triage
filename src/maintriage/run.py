"""End-to-end experiment: baselines, zero-shot LLM, QLoRA fine-tuning, evaluation.

    python -m maintriage.run --out results            # everything (needs a CUDA GPU)
    python -m maintriage.run --out results --cpu-only # data checks + classical baselines only

Every model is scored on the same held-out test split with the same metrics. Results, per-class
scores, predictions and the software environment are written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from importlib import metadata
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from maintriage.baseline import build_baseline
from maintriage.data import Splits, load_splits
from maintriage.metrics import score

PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "scikit-learn",
]


def _environment() -> dict:
    env = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            env[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env[name] = None
    try:
        import torch

        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["gpu"] = None
    return env


def _dataset_summary(splits: Splits) -> dict:
    return {
        "components": len(splits.labels),
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
        "test_class_counts": dict(Counter(splits.test["component"]).most_common()),
    }


def run(out: Path, cpu_only: bool, top_k: int, seed: int, epochs: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(top_k=top_k, seed=seed)
    gold = splits.test["component"].tolist()
    faults = splits.test["fault"].tolist()
    predictions = pd.DataFrame({"fault": faults, "gold": gold})
    results: dict = {"dataset": _dataset_summary(splits), "models": {}}

    majority = Counter(splits.train["component"]).most_common(1)[0][0]
    results["models"]["majority_class"] = score(gold, [majority] * len(gold))

    baseline = build_baseline(seed).fit(splits.train["fault"], splits.train["component"])
    predictions["tfidf_logreg"] = baseline.predict(faults)
    results["models"]["tfidf_logreg"] = score(gold, predictions["tfidf_logreg"])

    if not cpu_only:
        import gc

        import torch

        from maintriage.llm import TrainConfig, finetune, load_model, load_tokenizer, predict

        config = TrainConfig(seed=seed, epochs=epochs)
        results["train_config"] = config.as_dict()
        tokenizer = load_tokenizer(config.model_id)

        base = load_model(config.model_id)
        predictions["zero_shot"], predictions["zero_shot_raw"] = predict(
            base, tokenizer, faults, splits.labels
        )
        results["models"]["zero_shot_qwen"] = score(gold, predictions["zero_shot"])
        del base
        gc.collect()
        torch.cuda.empty_cache()

        adapter_dir = str(out / "adapter")
        results["training"] = finetune(
            config,
            (splits.train["fault"].tolist(), splits.train["component"].tolist()),
            (splits.validation["fault"].tolist(), splits.validation["component"].tolist()),
            splits.labels,
            adapter_dir,
        )
        tuned = load_model(config.model_id, adapter_dir)
        predictions["qlora"], predictions["qlora_raw"] = predict(
            tuned, tokenizer, faults, splits.labels
        )
        results["models"]["qlora_qwen"] = score(gold, predictions["qlora"])

    best = max(
        (name for name in predictions.columns if name in {"tfidf_logreg", "zero_shot", "qlora"}),
        key=lambda name: score(gold, predictions[name])["macro_f1"],
    )
    per_class = classification_report(
        gold, predictions[best], labels=splits.labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(per_class).T.round(3).to_csv(out / f"per_class_{best}.csv")
    predictions.to_csv(out / "test_predictions.csv", index=False)
    results["environment"] = _environment()
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--cpu-only", action="store_true", help="skip the LLM stages")
    parser.add_argument("--top-k", type=int, default=20, help="number of components to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    results = run(args.out, args.cpu_only, args.top_k, args.seed, args.epochs)
    print(json.dumps(results["models"], indent=2))


if __name__ == "__main__":
    main()
