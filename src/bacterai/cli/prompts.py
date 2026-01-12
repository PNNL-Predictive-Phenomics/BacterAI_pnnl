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
