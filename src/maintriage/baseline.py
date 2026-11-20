"""Classical baseline: TF-IDF features + logistic regression.

Any fine-tuned LLM has to beat this cheap model to justify its GPU cost. Character n-grams are
used alongside words because the logs are full of typos, codes and run-together words
("faultat", "A/C", "TIMS").
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_baseline(seed: int = 42) -> Pipeline:
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=10.0, max_iter=2000, class_weight="balanced", random_state=seed
    )
    return Pipeline([("features", features), ("classifier", classifier)])
