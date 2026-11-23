"""rank_labels must equal the log-probability computed the slow, obvious way."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from maintriage.llm import balance_classes, rank_labels  # noqa: E402
from maintriage.prompts import messages  # noqa: E402

TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tiny():
    """A randomly initialised 2-layer Qwen2 model with the real Qwen tokenizer: no weights to
    download, but the same vocabulary and chat template as the experiment."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except OSError:
        pytest.skip("tokenizer not downloadable here")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return transformers.Qwen2ForCausalLM(config).eval(), tokenizer


def test_rank_labels_matches_full_forward(tiny):
    model, tokenizer = tiny
    labels = ["ACU", "Door Roller", "Saloon Light Ballast"]
    conversation = messages("door will not close", labels)
    scores = rank_labels(model, tokenizer, [conversation], labels)
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for col, label in enumerate(labels):
        answer = tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + answer])
        with torch.no_grad():
            logprobs = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
        expected = sum(
            logprobs[len(prompt_ids) - 1 + j, token].item() for j, token in enumerate(answer)
        )
        assert scores[0, col] == pytest.approx(expected, abs=1e-3)


def test_balance_classes_oversamples_to_median():
    faults = [f"r{i}" for i in range(9)]
    components = ["A"] * 5 + ["B"] * 3 + ["C"]
    out_faults, out_components = balance_classes(faults, components, seed=0)
    counts = {c: out_components.count(c) for c in "ABC"}
    assert counts == {"A": 5, "B": 3, "C": 3}  # median count is 3
    assert set(out_faults) == set(faults)
