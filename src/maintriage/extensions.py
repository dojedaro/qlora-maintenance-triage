"""Follow-up experiments on the same split: top-k suggestions, retrieval few-shot, class balance.

    python -m maintriage.extensions --out results/extensions   # needs a CUDA GPU, ~60 minutes

Questions this answers, all on the same 740 held-out reports as the main run:

1. **Top-k:** if technicians see the top 3 suggestions, how often is the right component there?
   Every model ranks all 20 components (LLMs by the probability of generating each name).
2. **Retrieval few-shot vs. fine-tuning:** the base model gets the 5 most similar past reports
   (TF-IDF nearest neighbours from the training set) with their repaired components. This is the
   cheap alternative to fine-tuning that a deployment team would try first.
3. **Class balance:** fine-tuning again with rare components oversampled to the median count, to
   see whether it fixes the components the first model never got right.
4. **Seed variance:** the original fine-tuning recipe rerun with a different seed.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from maintriage.baseline import build_baseline
from maintriage.data import load_splits
from maintriage.llm import TrainConfig, finetune, load_model, load_tokenizer, predict, rank_labels
from maintriage.metrics import score
from maintriage.prompts import messages

TOP_K = (1, 3)


def topk_accuracy(gold: list[str], scores: np.ndarray, labels: list[str], k: int) -> float:
    top = np.argsort(-scores, axis=1)[:, :k]
    index = {label: i for i, label in enumerate(labels)}
    return round(float(np.mean([index[g] in row for g, row in zip(gold, top, strict=True)])), 4)


def ranked_summary(gold: list[str], scores: np.ndarray, labels: list[str]) -> dict:
    best = [labels[i] for i in scores.argmax(axis=1)]
    return {
        **{f"top{k}_accuracy": topk_accuracy(gold, scores, labels, k) for k in TOP_K},
        "top1_macro_f1": round(
            float(f1_score(gold, best, labels=labels, average="macro", zero_division=0)), 4
        ),
    }


def nearest_examples(
    train_faults: list[str], train_components: list[str], queries: list[str], k: int = 5
) -> list[list[tuple[str, str]]]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    train_matrix = vectorizer.fit_transform(train_faults)
    similarity = (vectorizer.transform(queries) @ train_matrix.T).toarray()
    nearest = np.argsort(-similarity, axis=1)[:, :k]
    return [[(train_faults[j], train_components[j]) for j in row] for row in nearest]


def _release_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run(out: Path, seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(seed=seed)
    labels = splits.labels
    gold = splits.test["component"].tolist()
    faults = splits.test["fault"].tolist()
    train = (splits.train["fault"].tolist(), splits.train["component"].tolist())
    validation = (splits.validation["fault"].tolist(), splits.validation["component"].tolist())
    results: dict = {"n_test": len(gold), "models": {}}
    predictions = pd.DataFrame({"fault": faults, "gold": gold})

    baseline = build_baseline(seed).fit(*train)
    probabilities = baseline.predict_proba(faults)
    order = [list(baseline.classes_).index(label) for label in labels]
    results["models"]["tfidf_logreg"] = ranked_summary(gold, probabilities[:, order], labels)

    # Retrieval alone: majority vote among the 5 most similar training reports.
    shots = nearest_examples(*train, faults)
    votes = [Counter(c for _, c in examples).most_common(1)[0][0] for examples in shots]
    results["models"]["knn5_vote"] = score(gold, votes)

    config = TrainConfig(seed=seed)
    tokenizer = load_tokenizer(config.model_id)
    base = load_model(config.model_id)

    zero = [messages(f, labels) for f in faults]
    results["models"]["zero_shot_ranked"] = ranked_summary(
        gold, rank_labels(base, tokenizer, zero, labels), labels
    )

    few = [messages(f, labels, examples) for f, examples in zip(faults, shots, strict=True)]
    few_scores = rank_labels(base, tokenizer, few, labels)
    results["models"]["retrieval_5shot_ranked"] = ranked_summary(gold, few_scores, labels)
    generated, raw = _predict_conversations(base, tokenizer, few, labels)
    predictions["retrieval_5shot"], predictions["retrieval_5shot_raw"] = generated, raw
    results["models"]["retrieval_5shot_generated"] = score(gold, generated)
    del base
    _release_gpu_memory()

    for name, run_config in (
        ("qlora_seed43", TrainConfig(seed=43)),
        ("qlora_balanced", TrainConfig(seed=seed, balance=True)),
    ):
        adapter = str(out / f"adapter_{name}")
        results[f"training_{name}"] = finetune(run_config, train, validation, labels, adapter)
        _release_gpu_memory()
        tuned = load_model(config.model_id, adapter)
        ranked = rank_labels(tuned, tokenizer, zero, labels)
        results["models"][f"{name}_ranked"] = ranked_summary(gold, ranked, labels)
        predictions[name], predictions[f"{name}_raw"] = predict(tuned, tokenizer, faults, labels)
        results["models"][f"{name}_generated"] = score(gold, predictions[name])
        del tuned
        _release_gpu_memory()

    predictions.to_csv(out / "test_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def _predict_conversations(model, tokenizer, conversations, labels, batch_size: int = 8):
    """Greedy generation for prebuilt conversations (the few-shot prompts are longer)."""
    from maintriage.prompts import parse_answer

    parsed, raw = [], []
    for start in range(0, len(conversations), batch_size):
        prompts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations[start : start + batch_size]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        answers = tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        raw.extend(answers)
        parsed.extend(parse_answer(a, labels) for a in answers)
    return parsed, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/extensions"))
    args = parser.parse_args()
    print(json.dumps(run(args.out)["models"], indent=2))


if __name__ == "__main__":
    main()
