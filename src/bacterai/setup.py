"""
Interactive experiment setup and file saving.
"""
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bacterai.configuration import ExperimentConfig, IngredientsConfig


def _prompt_nonempty(message: str) -> str:
    while True:
        resp = input(message).strip()
        if resp:
            return resp
        print("Input cannot be empty. Please try again.")


def _prompt_yes_no(message: str) -> bool:
    choices = {"y": True, "Y": True, "n": False, "N": False}
    while True:
        resp = input(message).strip()
        if resp in choices:
            return choices[resp]
        print("Invalid input. Please enter Y or N.")


def _expand_to_abs(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _scan_next_index(base_dir: Path) -> int:
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


def _ensure_experiment_dir(base_dir: Path, start_idx: int, dry_run: bool, fixed_path: Optional[Path] = None) -> Tuple[Optional[Path], int]:
    """
    Ensure an experiment directory exists.
    If fixed_path is provided, uses that exact path instead of experiment_N pattern.
    If directory exists, prompts user to overwrite, provide new path, or skip (or try next index if not fixed_path).
    Returns (directory_path, index_used) or (None, index) if user skips.
    """
    idx = start_idx
    while True:
        exp_dir = fixed_path if fixed_path else base_dir / f"experiment_{idx}"
        
        if not exp_dir.exists():
            if not dry_run:
                exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir, idx
        
        overwrite = _prompt_yes_no(f"Directory {exp_dir} already exists. Overwrite entire folder? (Y/N): ")
        if overwrite:
            if not dry_run:
                shutil.rmtree(exp_dir)
                exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir, idx
        
        # Ask if user wants to provide a custom path instead
        use_new = _prompt_yes_no("Would you like to provide a different experiment path? (Y/N): ")
        if use_new:
            new_path_str = _prompt_nonempty("Enter new experiment path: ")
            new_exp_dir = _expand_to_abs(new_path_str)
            if new_exp_dir.exists():
                overwrite_new = _prompt_yes_no(f"Directory {new_exp_dir} already exists. Overwrite entire folder? (Y/N): ")
                if not overwrite_new:
                    if fixed_path:
                        # For fixed paths, loop back to ask for another path
                        continue
                    else:
                        # For numbered paths, try next increment
                        idx += 1
                        continue
                if not dry_run:
                    shutil.rmtree(new_exp_dir)
            if not dry_run:
                new_exp_dir.mkdir(parents=True, exist_ok=True)
            return new_exp_dir, idx
        
        # Ask if user wants to skip
        skip = _prompt_yes_no("Skip this experiment? (Y/N): ")
        if skip:
            return None, idx
        
        # If not fixed_path, try next index; otherwise loop back
        if not fixed_path:
            idx += 1
        # If fixed_path, loop continues to ask again


def _write_json(path: Path, payload: Any, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write: {path}")
        print(json.dumps(payload, indent=2))
    else:
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote: {path}")


def setup_experiments(
    input_path: str,
    sheet: Optional[Union[str, int]],
    force_format: Optional[str],
    outfile_name: str = "config.json",
    dry_run: bool = False,
) -> None:
    """
    Orchestrate experiment setup:
      - Write configs for rows with experiment_path.
      - Prompt and create experiment_n directories for rows missing experiment_path.
      - Prompt for ingredients source file and save as ingredients.json.
      - Update config with ingredients_file if missing.
    """
    configs = ExperimentConfig.parse_configs_lenient(
        input_path=input_path,
        sheet=sheet,
        force_format=force_format,
    )

    with_path: List[Dict[str, Any]] = [c for c in configs if c.get("experiment_path")]
    missing_path: List[Dict[str, Any]] = [c for c in configs if not c.get("experiment_path")]

    # Collect all configs with their paths for ingredients processing
    all_configs_with_paths: List[tuple[Dict[str, Any], Path]] = []

    # Process configs with experiment_path
    for cfg in with_path:
        exp_dir = _expand_to_abs(cfg["experiment_path"])
        
        # Use _ensure_experiment_dir with fixed_path to handle existing directories
        result_dir, _ = _ensure_experiment_dir(exp_dir.parent, 0, dry_run, fixed_path=exp_dir)
        if result_dir is None:
            # User chose to skip this experiment
            continue
        
        # Update config with potentially new path
        cfg["experiment_path"] = str(result_dir)
        exp_dir = result_dir
        
        # Ensure ingredients_file is set to default if missing
        if not cfg.get("ingredients_file"):
            cfg["ingredients_file"] = "ingredients.json"
        
        # Add to list for ingredients processing
        all_configs_with_paths.append((cfg, exp_dir))
        
        out_path = exp_dir / outfile_name
        _write_json(out_path, cfg, dry_run=dry_run)

    if not missing_path and not all_configs_with_paths:
        return

    # Interactive setup for missing experiment_path
    if missing_path:
        base_dir_str = _prompt_nonempty("Enter a base directory for experiments (e.g., ~/Desktop/experiments): ")
        base_dir = _expand_to_abs(base_dir_str)
        if not dry_run and not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)

        # Determine starting index under base_dir
        next_idx = _scan_next_index(base_dir)

        # Assign directories and write configs
        for cfg in missing_path:
            exp_dir, used_idx = _ensure_experiment_dir(base_dir, next_idx, dry_run=dry_run)
            if exp_dir is None:
                # User chose to skip this experiment
                next_idx = used_idx + 1
                continue
            next_idx = used_idx + 1
            # Update the config with the experiment_path
            cfg["experiment_path"] = str(exp_dir)
            
            # Ensure ingredients_file is set to default if missing
            if not cfg.get("ingredients_file"):
                cfg["ingredients_file"] = "ingredients.json"
            
            # Add to list for ingredients processing
            all_configs_with_paths.append((cfg, exp_dir))
            
            out_path = exp_dir / outfile_name
            _write_json(out_path, cfg, dry_run=dry_run)

    # Ingredients handling - always prompt for source ingredients file
    if not all_configs_with_paths:
        return
    
    if len(all_configs_with_paths) > 1:
        same_ing = _prompt_yes_no("Use the same ingredients file for all experiments? (Y/N): ")
    else:
        same_ing = True

    if same_ing:
        ing_path_str = _prompt_nonempty("Enter ingredients file path (CSV/XLSX): ")
        ing_sheet_raw = input("Enter Excel sheet name or index (optional, press Enter to skip): ").strip()
        ing_sheet: Optional[Union[str, int]] = None
        if ing_sheet_raw:
            ing_sheet = int(ing_sheet_raw) if ing_sheet_raw.isdigit() else ing_sheet_raw
        result = IngredientsConfig.parse_ingredients(
            input_path=_expand_to_abs(ing_path_str).as_posix(),
            sheet=ing_sheet,
            id_map_path=None,
        )
        
        # Save ingredients.json for each experiment
        ingredients_filename = "ingredients.json"
        for cfg, exp_dir in all_configs_with_paths:
            # Ensure ingredients_file is set in config (should already be set, but double-check)
            if not cfg.get("ingredients_file"):
                cfg["ingredients_file"] = ingredients_filename
                # Re-write config.json with updated ingredients_file
                out_path = exp_dir / outfile_name
                _write_json(out_path, cfg, dry_run=dry_run)
            
            # Write ingredients.json
            ingredients_out = exp_dir / ingredients_filename
            _write_json(ingredients_out, result, dry_run=dry_run)
    else:
        for cfg, exp_dir in all_configs_with_paths:
            print(f"\nExperiment directory: {exp_dir}")
            ing_path_str = _prompt_nonempty("Enter ingredients file path (CSV/XLSX) for this experiment: ")
            ing_sheet_raw = input("Enter Excel sheet name or index (optional, press Enter to skip): ").strip()
            ing_sheet: Optional[Union[str, int]] = None
            if ing_sheet_raw:
                ing_sheet = int(ing_sheet_raw) if ing_sheet_raw.isdigit() else ing_sheet_raw
            result = IngredientsConfig.parse_ingredients(
                input_path=_expand_to_abs(ing_path_str).as_posix(),
                sheet=ing_sheet,
                id_map_path=None,
            )
            
            # Ensure ingredients_file is set in config (should already be set, but double-check)
            ingredients_filename = "ingredients.json"
            if not cfg.get("ingredients_file"):
                cfg["ingredients_file"] = ingredients_filename
                # Re-write config.json with updated ingredients_file
                out_path = exp_dir / outfile_name
                _write_json(out_path, cfg, dry_run=dry_run)
            
            # Write ingredients.json
            ingredients_out = exp_dir / ingredients_filename
            _write_json(ingredients_out, result, dry_run=dry_run)