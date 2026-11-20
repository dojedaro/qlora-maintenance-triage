"""Loading a 4-bit model, batched greedy prediction, and QLoRA fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

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

    def as_dict(self) -> dict:
        return asdict(self)


def _compute_dtype() -> torch.dtype:
    # T4 GPUs (free Colab) have no bfloat16 support; newer GPUs do and train more stably with it.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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
    return parsed, raw


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
        prompt_ids = prompt_ids[: max_length - len(answer_ids)]
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
    tokenizer.padding_side = "right"  # right padding for training; generation switches back
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
        warmup_ratio=0.05,
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
