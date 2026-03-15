import os
from dotenv import load_dotenv

load_dotenv()

# run mode, tinker or local
RUN_MODE = os.getenv("RUN_MODE", "").lower()
# number of prompt optimzation hypothesis for each eval
NUM_HYPOTHESIS = int(os.getenv("NUM_HYPOTHESIS", 1))

def get_engine():
    if RUN_MODE == "tinker":
        from src.providers.tinker import TinkerProvider
        # gotta have that tinker api key ready
        return TinkerProvider(api_key=os.getenv("TINKER_API_KEY"))
    elif RUN_MODE == "local": # for a 5070 or Blackwell cluster:'
        from src.providers.vllm import VllmProvider
        # TODO
        return VllmProvider()# define the vllm server settings here (get them from your cluster/ the env file)
    else:
        raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}, please choose from 'tinker' or 'local'")