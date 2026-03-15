import os
from dotenv import load_dotenv

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "tinker").lower()

def get_engine():
    if RUN_MODE == "tinker":
        from src.providers.tinker import TinkerProvider
        # gotta have that tinker api key ready
        return TinkerProvider(api_key=os.getenv("TINKER_API_KEY"))
    else: # RUN_MODE is 'cluster' or '5070'
        from src.providers.vllm import VllmProvider
        return VllmProvider()

# our singleton engine used for running evals and self-distillation
engine = get_engine()