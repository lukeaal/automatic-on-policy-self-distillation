"""vLLM wrapper utilities for local GPU serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


class VLLMModel:
    """Manage a vLLM model instance for local GPU inference."""

    def __init__(self) -> None:
        self._llm: LLM | None = None

    def load_model_for_serving(
        self,
        model: str,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ) -> LLM:
        """Load a specified model across GPUs for serving."""
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Install it with `uv add vllm` to use local serving."
            ) from exc

        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
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
                "vLLM is not installed. Install it with `uv add vllm` to use local serving."
            ) from exc

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
