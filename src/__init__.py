"""Source package for automatic-on-policy-self-distillation."""

__all__ = ["agent", "cli", "config", "data", "eval", "models", "train", "providers"]

def __getattr__(name):
    if name in __all__:
        import importlib
        # relative import '.' ensures it stays within the package
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")