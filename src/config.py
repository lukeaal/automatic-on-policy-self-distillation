import os
from dotenv import load_dotenv

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "").lower()

def get_engine():
    if RUN_MODE == "tinker":
        from src.providers.tinker import TinkerProvider
        # gotta have that tinker api key ready
        return TinkerProvider(api_key=os.getenv("TINKER_API_KEY"))
    elif RUN_MODE == "local": # for a 5070 or Blackwell cluster:'
        from src.providers.vllm import VllmProvider
        return VllmProvider()
    else:
        raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}, please choose from 'tinker' or 'local'")

# our singleton engine used for running evals and self-distillation
engine = get_engine()