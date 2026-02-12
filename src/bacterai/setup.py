"""
Pure business logic for experiment setup and file operations.
No user interaction - all prompting moved to cli.py.
"""
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bacterai.configuration import ExperimentConfig, IngredientsConfig


# Constants
INGREDIENTS_FILENAME = "ingredients.json"


def scan_next_index(base_dir: Path) -> int:
    """Scan directory for next available experiment index."""
    pattern = re.compile(r"^experiment_(\d+)$")
    max_idx = 0
    if base_dir.exists():
        for child in base_dir.iterdir():
            if child.is_dir():
                m = pattern.match(child.name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        pass
    return max_idx + 1


def write_json(path: Path, payload: Any) -> None:
    """Write JSON data to file and return message strings."""
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    
    path_message = (f"Wrote: {path}")
    verbose_message = (path_message + f"\nContents of {path}:\n{json.dumps(payload, indent=2)}\n")
    
    return path_message, verbose_message


def write_experiment_files(
    experiments: List[Tuple[Dict[str, Any], Path]],
    shared_ingredients: Optional[Dict[str, Any]],
    individual_ingredients: Dict[Path, Dict[str, Any]],
    outfile_name: str = "config.json",
) -> Tuple[str, str, str, str]:
    """
    Write config.json and ingredients.json files for each experiment.
    Pure function with no user interaction.
    Returns (config_paths, config_verbose, ingredients_paths, ingredients_verbose)
    """
    use_shared_ingredients = shared_ingredients is not None
    
    config_paths = []
    config_verbose_lines = []
    ingredients_paths = []
    ingredients_verbose_lines = []
    
    for cfg, exp_dir in experiments:
        # Update config with path and ingredients file
        cfg["experiment_path"] = str(exp_dir)
        cfg["ingredients_file"] = INGREDIENTS_FILENAME
        
        # Write config.json
        config_path = exp_dir / outfile_name
        config_p, config_verbose = write_json(config_path, cfg)
        config_paths.append(config_p)
        config_verbose_lines.append(config_verbose)
        
        # Write ingredients.json
        ingredients_data = shared_ingredients if use_shared_ingredients else individual_ingredients[exp_dir]
        ingredients_path = exp_dir / INGREDIENTS_FILENAME
        ingredients_p, ingredients_verbose = write_json(ingredients_path, ingredients_data)
        ingredients_paths.append(ingredients_p)
        ingredients_verbose_lines.append(ingredients_verbose)

    # Combine all messages
    config_paths_str = "\n".join(config_paths)
    config_verbose_str = "\n".join(config_verbose_lines)
    ingredients_paths_str = "\n".join(ingredients_paths)
    ingredients_verbose_str = "\n".join(ingredients_verbose_lines)

    return config_paths_str, config_verbose_str, ingredients_paths_str, ingredients_verbose_str

