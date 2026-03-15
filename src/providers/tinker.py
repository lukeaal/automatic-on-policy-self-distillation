import asyncio
import tinker
from tinker import types
from .base import BaseProvider, CompletionResponse

# model default is llama 3.2 1b, this is the cheapest to sample from
class TinkerProvider(BaseProvider):  # base provider only has generate for now
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B", api_key: str = None):
        # API key via env var or explicit pass
        self.client = tinker.ServiceClient(api_key=api_key).create_sampling_client(base_model=model_id)
        self.tokenizer = self.client.get_tokenizer()

    async def generate(self, prompts: list[str], **kwargs) -> list[CompletionResponse]:
        """
        sends a batch of prompts to Tinker and awaits the results.
        """
        # prepare sampling params
        params = types.SamplingParams(
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            stop=kwargs.get("stop", [])
        )

        # run all requests concurrently
        futures = []
        for prompt in prompts:
            model_input = types.ModelInput.from_ints(self.tokenizer.encode(prompt))
            # sample_async returns a Future immediately
            futures.append(self.client.sample_async(
                prompt=model_input, 
                sampling_params=params, 
                num_samples=1
            ))

        
        results = []
        for future in futures:
            sample_response = await future 
            text = self.tokenizer.decode(sample_response.sequences[0].tokens)
            results.append(CompletionResponse(text=text))
            
        return results