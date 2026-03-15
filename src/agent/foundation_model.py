"""
Foundation model wrapper for OpenAI via LiteLLM.
The foundation model is used to generate hypothesis for prompt improvements.
"""

import os

from litellm import completion


class FoundationModel:
    """Simple OpenAI model connection using LiteLLM."""

    def __init__(
        self,
        model_id: str = "openai/gpt-5.4",
        api_key: str | None = None,
        reasoning_effort: str = "high",
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.reasoning_effort = reasoning_effort
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Pass api_key or define it in the environment."
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text completion for a single prompt."""
        kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        response = completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            **kwargs,
        )
        return response.choices[0].message.content
