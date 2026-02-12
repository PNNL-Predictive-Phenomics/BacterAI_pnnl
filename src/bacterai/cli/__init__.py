"""
CLI module for BacterAI.
Provides command handlers and user interaction functions.

Note: Exports are lazily imported to avoid circular imports.
"""

__all__ = [
    # Commands
    "experiment",
    "ingredients",
    "setup",
    "run",
    "process_data",
    # Prompts
    "prompt_nonempty",
    "prompt_yes_no",
    "prompt_for_ingredients",
    "handle_existing_directory",
]


def __getattr__(name):
    if name in {"experiment", "ingredients", "setup", "run", "process_data"}:
        from . import commands as _commands
        return getattr(_commands, name)
    if name in {"prompt_nonempty", "prompt_yes_no", "prompt_for_ingredients", "handle_existing_directory"}:
        from . import prompts as _prompts
        return getattr(_prompts, name)
    raise AttributeError(f"module 'bacterai.cli' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
