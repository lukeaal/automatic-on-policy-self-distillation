"""Frontier model module."""
from litellm import completion



class FrontierModel:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def form_new_hypothesis(self, prompt: str):
    def generate(self, prompt: str):
        return self.model.generate(prompt)


def create_eval_score_json(past_hypothesis: dict[str:float]) -> str:
    pass