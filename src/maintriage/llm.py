"""Loading a 4-bit model, batched greedy prediction, and QLoRA fine-tuning."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from maintriage.prompts import messages, parse_answer


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 512
    seed: int = 42
    # Oversample rare components up to the median component count, so the model is not trained
    # to always answer the most common repair for an ambiguous symptom.
    balance: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _compute_dtype() -> torch.dtype:
    # bfloat16 needs compute capability 8.0+ (A100, L4, H100). A free-Colab T4 is 7.5: there
    # torch may report bf16 as "supported" through slow emulation, so check the hardware instead.
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # decoder-only models must be left-padded for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, adapter_dir: str | None = None) -> PreTrainedModel:
    """Base model in 4-bit NF4 (QLoRA's storage format), optionally with trained LoRA adapters."""
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_compute_dtype(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization, device_map="auto"
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


@torch.inference_mode()
def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    faults: Sequence[str],
    labels: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: int = 16,
) -> tuple[list[str], list[str]]:
    """Greedy-decode an answer for each fault. Returns (parsed labels, raw model outputs)."""
    parsed, raw = [], []
    for start in range(0, len(faults), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                messages(fault, labels), tokenize=False, add_generation_prompt=True
            )
            for fault in faults[start : start + batch_size]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        answers = tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        raw.extend(answers)
        parsed.extend(parse_answer(answer, labels) for answer in answers)
        done = min(start + batch_size, len(faults))
        if done % (batch_size * 10) == 0 or done == len(faults):
            print(f"predicted {done}/{len(faults)}", flush=True)
    return parsed, raw


@torch.inference_mode()
def rank_labels(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    conversations: Sequence[list[dict[str, str]]],
    labels: Sequence[str],
) -> np.ndarray:
    """Log-probability that the model answers with each label, for each conversation.

    Returns an array of shape (conversations, labels): the summed log-probability of the label's
    tokens plus the end-of-turn token, i.e. the probability of generating exactly that answer.
    Ranking by it gives top-k suggestions and an answer that can never fall outside the list.
    """
    answer_ids = [
        tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        for label in labels
    ]
    longest = max(map(len, answer_ids))
    pad = tokenizer.pad_token_id
    scores = np.zeros((len(conversations), len(labels)), dtype=np.float32)
    for row, conversation in enumerate(conversations):
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # One batch per report: the same prompt followed by each candidate answer, right-padded.
        batch = [prompt_ids + ids + [pad] * (longest - len(ids)) for ids in answer_ids]
        mask = [
            [1] * (len(prompt_ids) + len(ids)) + [0] * (longest - len(ids)) for ids in answer_ids
        ]
        input_ids = torch.tensor(batch, device=model.device)
        # Only the logits that predict answer tokens are needed (saves memory on a 150k vocab).
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.tensor(mask, device=model.device),
            logits_to_keep=longest + 1,
        ).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)  # position j predicts answer token j
        for col, ids in enumerate(answer_ids):
            target = torch.tensor(ids, device=model.device)
            scores[row, col] = logprobs[col, torch.arange(len(ids)), target].sum().item()
        if (row + 1) % 100 == 0 or row + 1 == len(conversations):
            print(f"ranked {row + 1}/{len(conversations)}", flush=True)
    return scores


def balance_classes(
    faults: Sequence[str], components: Sequence[str], seed: int
) -> tuple[list[str], list[str]]:
    """Repeat examples of components below the median count until they reach it."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {}
    for fault, component in zip(faults, components, strict=True):
        by_class.setdefault(component, []).append(fault)
    target = int(np.median([len(v) for v in by_class.values()]))
    out_faults, out_components = list(faults), list(components)
    for component, items in by_class.items():
        extra = max(0, target - len(items))
        for i in rng.choice(len(items), extra, replace=True) if extra else []:
            out_faults.append(items[i])
            out_components.append(component)
    return out_faults, out_components


def _warmup_steps(n_examples: int, config: TrainConfig, fraction: float = 0.05) -> int:
    steps_per_epoch = math.ceil(n_examples / (config.batch_size * config.grad_accum))
    return max(1, int(fraction * steps_per_epoch * config.epochs))


def _tokenize_examples(
    faults: Sequence[str],
    components: Sequence[str],
    labels: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Prompt + answer token ids, with the loss masked (-100) on the prompt.

    The model is only trained to produce the component name, not to reproduce the prompt.
    """
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for fault, component in zip(faults, components, strict=True):
        prompt = tokenizer.apply_chat_template(
            messages(fault, labels), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(component + tokenizer.eos_token, add_special_tokens=False)[
            "input_ids"
        ]
        if len(prompt_ids) + len(answer_ids) > max_length:
            # Truncating would cut the assistant header and teach the wrong format.
            raise ValueError(
                f"example of {len(prompt_ids) + len(answer_ids)} tokens > {max_length}"
            )
        ids = prompt_ids + answer_ids
        rows["input_ids"].append(ids)
        rows["attention_mask"].append([1] * len(ids))
        rows["labels"].append([-100] * len(prompt_ids) + answer_ids)
    return Dataset.from_dict(rows)


def finetune(
    config: TrainConfig,
    train: tuple[Sequence[str], Sequence[str]],
    validation: tuple[Sequence[str], Sequence[str]],
    labels: Sequence[str],
    output_dir: str,
) -> dict:
    """QLoRA: frozen 4-bit base model, trainable low-rank adapters on every linear layer.

    Returns training statistics (time, peak GPU memory, trainable parameters, losses).
    """
    torch.manual_seed(config.seed)
    tokenizer = load_tokenizer(config.model_id)
    tokenizer.padding_side = "right"  # this tokenizer is only used for training batches
    model = prepare_model_for_kbit_training(load_model(config.model_id))
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config.balance:
        train = balance_classes(*train, seed=config.seed)
    train_ds = _tokenize_examples(*train, labels, tokenizer, config.max_length)
    val_ds = _tokenize_examples(*validation, labels, tokenizer, config.max_length)
    dtype = _compute_dtype()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        # warmup_ratio was removed in transformers 5; an explicit step count works everywhere.
        warmup_steps=_warmup_steps(len(train_ds), config),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        report_to="none",
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - started
    model.save_pretrained(output_dir)

    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    return {
        "train_seconds": round(seconds, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "trainable_parameters": trainable,
        "train_loss": round(result.training_loss, 4),
        "validation_loss_per_epoch": [round(x, 4) for x in eval_losses],
        "train_examples": len(train_ds),
    }
