"""Training examples must match what the model sees at generation time, and only the answer may
carry loss. Uses the real Qwen tokenizer (a small download); skipped if transformers is absent."""

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from maintriage.llm import _tokenize_examples, load_tokenizer  # noqa: E402
from maintriage.prompts import messages  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["ACU", "Door Roller", "Saloon Light"]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return load_tokenizer(MODEL_ID)
    except OSError:
        pytest.skip("tokenizer not downloadable here")


def test_loss_covers_only_the_answer_and_end_token(tokenizer):
    row = _tokenize_examples(["Door B1 will not close"], ["Door Roller"], LABELS, tokenizer, 512)[0]
    supervised = [
        t for t, label in zip(row["input_ids"], row["labels"], strict=True) if label != -100
    ]
    assert tokenizer.decode(supervised) == "Door Roller" + tokenizer.eos_token


def test_training_prompt_matches_generation_prompt(tokenizer):
    fault = "TC reported saloon light not working at car 3540"
    row = _tokenize_examples([fault], ["Saloon Light"], LABELS, tokenizer, 512)[0]
    prompt = tokenizer.apply_chat_template(
        messages(fault, LABELS), tokenize=False, add_generation_prompt=True
    )
    generation_ids = tokenizer(prompt)["input_ids"]
    assert row["input_ids"][: len(generation_ids)] == generation_ids


def test_overlong_examples_are_rejected_not_truncated(tokenizer):
    with pytest.raises(ValueError):
        _tokenize_examples(["x " * 600], ["ACU"], LABELS, tokenizer, 512)
