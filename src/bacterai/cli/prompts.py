"""
User interaction and prompting functions for the CLI.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from bacterai.configuration import IngredientsConfig


def prompt_nonempty(message: str) -> str:
    """Prompt user for non-empty input."""
    while True:
        resp = input(message).strip()
        if resp:
            return resp
        print("Input cannot be empty. Please try again.")


def prompt_yes_no(message: str) -> bool:
    """Prompt user for yes/no input."""
    choices = {"y": True, "Y": True, "n": False, "N": False}
    while True:
        resp = input(message).strip()
        if resp in choices:
            return choices[resp]
        print("Invalid input. Please enter Y or N.")


def prompt_missing_experiment_path_source() -> str:
    """Prompt for how to resolve a missing experiment path."""
    use_current = prompt_yes_no(
        "No experiment path was provided. Use the current working directory? (Y/N): "
    )
    return "current" if use_current else "full_path"


def prompt_experiment_folder_name() -> str:
    """Prompt for a folder name (single path segment) when using current location."""
    while True:
        folder_name = prompt_nonempty("Enter experiment folder name: ")
        if Path(folder_name).is_absolute():
            print("Please provide only a folder name, not a full path.")
            continue
        if "/" in folder_name or "\\" in folder_name:
            print("Please provide only a folder name (no path separators).")
            continue
        return folder_name


def prompt_full_experiment_path() -> Path:
    """Prompt for a full experiment folder path and validate it includes a folder name."""
    while True:
        raw_path = prompt_nonempty(
            "Enter full experiment folder path (include experiment folder name): "
        )
        path = Path(raw_path).expanduser().resolve()
        if path.parent == path or not path.name:
            print("Path must include a folder name (not only a root path).")
            continue
        return path


def prompt_overwrite_nonempty_directory(exp_dir: Path) -> bool:
    """Prompt whether to overwrite a non-empty directory."""
    return prompt_yes_no(
        f"Directory {exp_dir} already exists and contains files. Overwrite entire folder? (Y/N): "
    )


def prompt_for_ingredients() -> Dict[str, Any]:
    """Prompt user for ingredients file and parse it."""
    ing_path_str = prompt_nonempty("Enter ingredients file path (CSV/XLSX): ")
    ing_sheet_raw = input("Enter Excel sheet name or index (optional, press Enter to skip): ").strip()
    ing_sheet: Optional[Union[str, int]] = None
    if ing_sheet_raw:
        ing_sheet = int(ing_sheet_raw) if ing_sheet_raw.isdigit() else ing_sheet_raw
    return IngredientsConfig.parse_ingredients(
        input_path=Path(ing_path_str).expanduser().resolve().as_posix(),
        sheet=ing_sheet,
        id_map_path=None,
    )


def handle_existing_directory(exp_dir: Path) -> str:
    """
    Handle what to do when a directory already exists.
    Returns: 'overwrite', 'skip', or 'use_new_path'
    """
    overwrite = prompt_yes_no(f"Directory {exp_dir} already exists. Overwrite entire folder? (Y/N): ")
    if overwrite:
        return 'overwrite'
    
    use_new = prompt_yes_no("Would you like to provide a different experiment path? (Y/N): ")
    if use_new:
        return 'use_new_path'
    
    skip = prompt_yes_no("Skip this experiment? (Y/N): ")
    if skip:
        return 'skip'
    
    # If they said no to everything, default to overwrite
    return 'overwrite'
