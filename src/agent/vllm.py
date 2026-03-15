"""vLLM wrapper utilities for local GPU serving."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


class VLLMModel:
    """Manage a vLLM model instance for local GPU inference."""

    def __init__(self) -> None:
        self._llm: LLM | None = None

    @staticmethod
    def _available_gpu_count() -> int:
        """Best-effort GPU count for vLLM tensor parallel sizing."""
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            visible = [device.strip() for device in cuda_visible_devices.split(",")]
            visible = [device for device in visible if device and device != "-1"]
            if visible:
                return len(visible)

        try:
            import torch

            return max(1, int(torch.cuda.device_count()))
        except Exception:
            return 1

    def load_model_for_serving(
        self,
        model: str,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ) -> LLM:
        """Load a specified model across GPUs for serving."""
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. On Linux, install it with `uv sync --group linux` "
                "to use local serving."
            ) from exc

        available_gpus = self._available_gpu_count()
        requested_parallelism = (
            available_gpus if tensor_parallel_size is None else max(1, tensor_parallel_size)
        )
        effective_parallelism = min(requested_parallelism, available_gpus)
        if effective_parallelism < requested_parallelism:
            warnings.warn(
                "Requested tensor_parallel_size="
                f"{requested_parallelism}, but only {available_gpus} GPU(s) are available. "
                f"Using tensor_parallel_size={effective_parallelism}.",
                stacklevel=2,
            )

        self._llm = LLM(
            model=model,
            tensor_parallel_size=effective_parallelism,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        return self._llm

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 100,
    ) -> list[str]:
        """Generate text for prompts using the loaded vLLM model."""
        if self._llm is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model_for_serving() before generate()."
            )

        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. On Linux, install it with `uv sync --group linux` "
                "to use local serving."
            ) from exc

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
