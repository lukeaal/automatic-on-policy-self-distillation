"""Source package for automatic-on-policy-self-distillation."""

from src import agent
from src import cli
from src import config
from src import data
from src import eval
from src import models
from src import train
from src import viz

__all__ = ["agent", "cli", "config", "data", "eval", "models", "train", "viz"]
