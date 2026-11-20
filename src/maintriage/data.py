"""Turn raw SMRT maintenance logs into a clean fault-triage dataset.

Each raw record is ``<fault title> ~~~~~ <fault details + corrective action>`` with a label naming
the component that was repaired. The task here is triage: predict the component from the fault
report alone, as a technician would before opening the panel. So the corrective action, which
usually names the replaced part, must never reach the model.

Pipeline: extract fault text -> cut anything written after the repair -> drop duplicates ->
keep the most frequent components -> stratified train/validation/test split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ID = "ASNIJKDNI/SMRT-Maintenance-Logs"
SEPARATOR = "~~~~~"

# Phrases that only appear once the repair is done. Text from the first match onward is dropped,
# because it usually names the part that was replaced (the label). Future-tense requests such as
# "to replace door roller" stay: that is the technician's diagnosis, available at triage time.
_AFTER_REPAIR = re.compile(
    r"\b(replaced|changed|renewed|rectified|retightened|tightened|adjusted|recharged|charged in"
    r"|cleaned|reset|done by|function(?:al)? test(?:ed)?|tested ok|test(?:ed)? - ?ok|carried out"
    r"|found|replace(?:ment)? done)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def duplicate_key(text: str) -> str:
    """Key under which two reports count as the same: case, punctuation and numbers ignored.

    Many reports are templates that differ only in the train or car number ("TC reported 3709
    internal smoke pre alarm"); treating those as one report keeps templates from landing on both
    sides of the split.
    """
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", _normalize(text))).strip()


def extract_fault(raw: str) -> str:
    """Return only the fault report from a raw log entry, with the corrective action removed.

    The field after the separator often restates the fault in full before the action, separated
    by two or more spaces. That fuller restatement is used when it begins like the title;
    otherwise the title alone is the safest fault text.
    """
    title, _, rest = raw.partition(SEPARATOR)
    title = title.strip()
    first, *tail = re.split(r"\s{2,}", rest.strip(), maxsplit=1)
    restates_title = _normalize(first)[:20] == _normalize(title)[:20]
    fault = first if tail and restates_title and len(first) >= len(title) else title

    match = _AFTER_REPAIR.search(fault)
    if match and match.start() > 0:
        fault = fault[: match.start()]
    return _WHITESPACE.sub(" ", fault).strip(" .,;:-")


def build_frame(raw: pd.DataFrame, label_names: list[str], min_chars: int = 12) -> pd.DataFrame:
    """Raw ``text``/``label`` rows -> one row per unique fault report with a component name."""
    frame = pd.DataFrame(
        {
            "fault": raw["text"].map(extract_fault),
            "component": raw["label"].map(lambda i: label_names[int(i)]),
        }
    )
    frame = frame[frame["fault"].str.len() >= min_chars].copy()
    frame["key"] = frame["fault"].map(duplicate_key)
    # Identical reports filed under different components cannot all be right; drop them rather
    # than keep whichever came first. Then keep one row per report so none can appear in both
    # train and test (the published splits share hundreds of rows).
    labels_per_key = frame.groupby("key")["component"].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index
    frame = frame[~frame["key"].isin(conflicting)]
    frame = frame.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    return frame


def top_components(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the k most frequent components. Ties are broken by name: pandas versions order tied
    counts differently, which would otherwise change the label set and the split."""
    counts = frame["component"].value_counts()
    ranked = sorted(counts.index, key=lambda name: (-counts[name], name))
    keep = ranked[:k]
    return frame[frame["component"].isin(keep)].reset_index(drop=True)


def split(
    frame: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.1, seed: int = 42
) -> Splits:
    """Stratified split so every component appears in train, validation and test."""
    train_val, test = train_test_split(
        frame, test_size=test_size, stratify=frame["component"], random_state=seed
    )
    train, validation = train_test_split(
        train_val,
        test_size=validation_size / (1 - test_size),
        stratify=train_val["component"],
        random_state=seed,
    )
    labels = sorted(frame["component"].unique())
    return Splits(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
        labels,
    )


def load_splits(top_k: int = 20, seed: int = 42) -> Splits:
    """Download the dataset from the Hugging Face Hub and return the triage splits."""
    from datasets import load_dataset  # imported lazily so the parsing code has no heavy deps

    dataset = load_dataset(DATASET_ID)
    label_names = dataset["train"].features["label"].names
    raw = pd.concat([dataset[name].to_pandas() for name in dataset], ignore_index=True)
    return split(top_components(build_frame(raw, label_names), top_k), seed=seed)
