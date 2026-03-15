"""Minimal on-policy self-distillation loop with reverse KL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import logging
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _load_rows_from_json_path(path: str) -> list[dict[str, object]]:
    """Load rows from either JSONL or regular JSON files."""
    with Path(path).open(encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        raise ValueError(f"Dataset file is empty: {path}")

    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    raise ValueError(f"Unsupported JSON dataset structure in {path}: expected a top-level list")


class SelfDistillDataset(Dataset):
    """Dataset of teacher/student prompts for on-policy distillation."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    @classmethod
    def from_source(cls, source: str | Path | Iterable[Mapping[str, object]]) -> "SelfDistillDataset":
        if isinstance(source, (str, Path)):
            path = str(source)
            if path.endswith(".json") or path.endswith(".jsonl"):
                rows = _load_rows_from_json_path(path)
            elif path.endswith(".parquet"):
                logger.info("Loading parquet dataset from %s", path)
                split = datasets.load_dataset("parquet", data_files=path)["train"]
                rows = [dict(row) for row in split]
            else:
                raise ValueError(f"Unsupported dataset path: {path}")
        else:
            rows = [dict(row) for row in source]

        normalized = []
        for row in rows:
            student_prompt = str(row["student_prompt"]).strip()
            teacher_prompt = str(row["teacher_prompt"]).strip()
            normalized.append(
                {
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                }
            )
        return cls(normalized)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def _collate_examples(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "student_prompt": [row["student_prompt"] for row in batch],
        "teacher_prompt": [row["teacher_prompt"] for row in batch],
    }


def load_model_from_weights(
    model_name_or_path: str,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and tokenizer from pretrained weights."""
    logger.info("Loading model and tokenizer from %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if device is not None:
        model.to(device)
    logger.info("Loaded model on device=%s", device if device is not None else "default")
    return model, tokenizer


def load_self_distill_dataloader(
    source: str | Path | Iterable[Mapping[str, object]],
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Load prompt pairs into a DataLoader."""
    dataset = SelfDistillDataset.from_source(source)
    logger.info("Loaded self-distill dataset with %d examples", len(dataset))
    logger.info("Creating dataloader batch_size=%d shuffle=%s", batch_size, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_examples)


def _build_sequence_tensors(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[list[int]],
    max_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    input_id_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    loss_mask_rows: list[list[int]] = []

    for prompt, response_ids in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        full_response_ids = list(response_ids)
        if eos_id is not None and (not full_response_ids or full_response_ids[-1] != eos_id):
            full_response_ids.append(eos_id)

        full_ids = (prompt_ids + full_response_ids)[:max_length]
        response_start = min(len(prompt_ids), max_length)
        loss_mask = [0] * response_start + [1] * max(0, len(full_ids) - response_start)

        pad_len = max_length - len(full_ids)
        input_id_rows.append(full_ids + [pad_id] * pad_len)
        attention_rows.append([1] * len(full_ids) + [0] * pad_len)
        loss_mask_rows.append(loss_mask + [0] * pad_len)

    input_ids = torch.tensor(input_id_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
    loss_mask = torch.tensor(loss_mask_rows, dtype=torch.bool, device=device)
    return input_ids, attention_mask, loss_mask


def _response_logits(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_loss_mask = loss_mask[:, 1:]
    return shift_logits[shift_loss_mask]


def compute_reverse_kl_loss(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    """Compute KL(student || teacher) over aligned response-token logits."""
    n = min(teacher_logits.size(0), student_logits.size(0))
    if n == 0:
        return student_logits.new_zeros(())

    teacher_log_probs = F.log_softmax(teacher_logits[:n].float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits[:n].float(), dim=-1)
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()


def training_loop(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1,
    max_new_tokens: int = 128,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    teacher_update_steps: int = 0,
) -> PreTrainedModel:
    """Run on-policy self-distillation with reverse KL."""
    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    total_steps = len(dataloader) * epochs
    logger.info(
        "Starting self-distillation epochs=%d total_steps=%d student_device=%s teacher_device=%s max_new_tokens=%d max_length=%d",
        epochs,
        total_steps,
        student_device,
        teacher_device,
        max_new_tokens,
        max_length,
    )

    global_step = 0
    for epoch in range(epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)
        for batch_idx, batch in enumerate(dataloader, start=1):
            student_model.train()
            logger.info(
                "Step %d/%d (epoch %d batch %d): generating rollouts for batch_size=%d",
                global_step + 1,
                total_steps,
                epoch + 1,
                batch_idx,
                len(batch["student_prompt"]),
            )
            prompts = tokenizer(
                batch["student_prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(student_device)

            with torch.no_grad():
                generated = student_model.generate(
                    **prompts,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response_ids = generated[:, prompts["input_ids"].size(1):].tolist()
            teacher_input_ids, teacher_attention_mask, teacher_loss_mask = _build_sequence_tensors(
                tokenizer=tokenizer,
                prompts=batch["teacher_prompt"],
                responses=response_ids,
                max_length=max_length,
                device=teacher_device,
            )
            student_input_ids, student_attention_mask, student_loss_mask = _build_sequence_tensors(
                tokenizer=tokenizer,
                prompts=batch["student_prompt"],
                responses=response_ids,
                max_length=max_length,
                device=student_device,
            )

            with torch.no_grad():
                teacher_logits = _response_logits(
                    teacher_model,
                    teacher_input_ids,
                    teacher_attention_mask,
                    teacher_loss_mask,
                )
                teacher_logits = teacher_logits.to(student_device)

            student_logits = _response_logits(
                student_model,
                student_input_ids,
                student_attention_mask,
                student_loss_mask,
            )

            loss = compute_reverse_kl_loss(teacher_logits, student_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            logger.info(
                "Step %d/%d complete: reverse_kl=%.6f response_tokens=%d",
                global_step,
                total_steps,
                loss.detach().item(),
                min(teacher_logits.size(0), student_logits.size(0)),
            )
            if teacher_update_steps > 0 and global_step % teacher_update_steps == 0:
                teacher_model.load_state_dict(student_model.state_dict())
                teacher_model.eval()
                logger.info("Updated teacher weights from student at step %d", global_step)

        logger.info("Finished epoch %d/%d", epoch + 1, epochs)

    logger.info("Self-distillation complete after %d steps", global_step)
    return student_model


def self_distill(
    model_name_or_path: str,
    dataset_source: str | Path | Iterable[Mapping[str, object]],
    batch_size: int = 4,
    epochs: int = 1,
    lr: float = 1e-5,
    max_new_tokens: int = 128,
    max_length: int = 512,
    teacher_update_steps: int = 0,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load models, data, and run reverse-KL self-distillation."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("self-distill requires CUDA and at least 2 visible GPUs, but CUDA is not available")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"self-distill requires at least 2 visible GPUs, but found {gpu_count}")

    student_device = torch.device("cuda:0")
    teacher_device = torch.device("cuda:1")
    logger.info(
        "Preparing self-distillation run model=%s student_device=%s teacher_device=%s",
        model_name_or_path,
        student_device,
        teacher_device,
    )
    student_model, tokenizer = load_model_from_weights(model_name_or_path, student_device)
    teacher_model, _ = load_model_from_weights(model_name_or_path, teacher_device)
    dataloader = load_self_distill_dataloader(dataset_source, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(student_model.parameters(), lr=lr)
    logger.info("Optimizer initialized with lr=%s", lr)

    trained_student = training_loop(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=epochs,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        teacher_update_steps=teacher_update_steps,
    )
    logger.info("Returning trained student model")
    return trained_student, tokenizer


def main() -> None:
    raise SystemExit("Call self_distill(...) from Python to run training.")


if __name__ == "__main__":
    main()
