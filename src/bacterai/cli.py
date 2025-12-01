"""
CLI-facing functions for the unified configuration tool.
All user interaction (printing, file writing decisions) happens here.
"""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from bacterai.configuration import ExperimentConfig, IngredientsConfig
from bacterai.setup import write_experiment_files, scan_next_index
from bacterai.run.core import execute_experiment
from bacterai.run.paths import get_round


# User interaction functions
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


def prepare_experiment_directories(configs: List[Dict[str, Any]]) -> List[tuple[Dict[str, Any], Path]]:
    """
    Prepare experiment directories with interactive prompts for existing directories.
    Returns list of (config, exp_dir) tuples ready for file writing.
    """
    experiments = []
    base_dir = None
    
    for cfg in configs:
        if cfg.get("experiment_path"):
            # Config has a path - use it
            exp_dir = Path(cfg["experiment_path"]).expanduser().resolve()
            
            # Handle existing directory
            if exp_dir.exists():
                action = handle_existing_directory(exp_dir)
                if action == 'skip':
                    continue
                elif action == 'use_new_path':
                    new_path_str = prompt_nonempty("Enter new experiment path: ")
                    exp_dir = Path(new_path_str).expanduser().resolve()
                    if exp_dir.exists():
                        action = handle_existing_directory(exp_dir)
                        if action == 'skip':
                            continue
                # If action is 'overwrite' or we fall through, continue to directory creation
            
            # Create/recreate directory
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
                
        else:
            # Config missing path - need base_dir
            if base_dir is None:
                base_dir_str = prompt_nonempty("Enter a base directory for experiments (e.g., ~/Desktop/experiments): ")
                base_dir = Path(base_dir_str).expanduser().resolve()
                if not base_dir.exists():
                    base_dir.mkdir(parents=True, exist_ok=True)
            
            # Find next available experiment directory
            next_idx = scan_next_index(base_dir)
            exp_dir = base_dir / f"experiment_{next_idx}"
            
            # Handle existing directory
            if exp_dir.exists():
                action = handle_existing_directory(exp_dir)
                if action == 'skip':
                    continue
                # For auto-generated paths, we don't offer 'use_new_path' since the path is auto-generated
                
            # Create/recreate directory
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
        
        experiments.append((cfg, exp_dir))
    
    return experiments


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


def experiment(
    input_path: str,
    sheet: Optional[Union[str, int]],
    outfile_name: str,
    force_format: Optional[str],
    verbose: bool,
) -> None:
    """
    Convert experiment settings (CSV/XLSX) to JSON file(s).
    Prints status messages and writes files. If verbose is True, also prints file contents.
    """
    cfgs = ExperimentConfig.parse_configs(
        input_path=input_path,
        sheet=sheet,
        force_format=force_format,
    )
    if not cfgs:
        print("No experiments to process.")
        return

    # Prepare experiment directories with interactive prompts
    experiments = prepare_experiment_directories(cfgs)
    if not experiments:
        print("No experiments to process.")
        return

    # Write config files
    written: List[str] = []
    for cfg, exp_dir in experiments:
        # Update config with path
        cfg["experiment_path"] = str(exp_dir)
        out_path = exp_dir / outfile_name
        
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        written.append(str(out_path))
        
        if verbose:
            print(f"Contents of {out_path}:")
            print(json.dumps(cfg, indent=2))
            print()

    if written:
        print("Wrote config file(s):")
        for p in written:
            print(f"  - {p}")
    else:
        print("No config files written.", file=sys.stderr)


def ingredients(
    input_path: str,
    sheet: Optional[Union[str, int]],
    output: Optional[str],
    id_map_path: Optional[str],
    verbose: bool = False,
) -> None:
    """
    Convert ingredient sheets (Format 1 or 2; CSV/XLSX) to a JSON payload.
    Writes to output if provided, otherwise prints to stdout.
    If verbose is True, also prints contents when writing to a file.
    """
    result = IngredientsConfig.parse_ingredients(
        input_path=input_path,
        sheet=sheet,
        id_map_path=id_map_path,
    )
    out_str = json.dumps(result, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(out_str)
        print(f"Wrote: {output}")
        if verbose:
            print(f"Contents of {output}:")
            print(out_str)
    else:
        print(out_str)


def setup(
    input_path: str,
    sheet: Optional[Union[str, int]],
    force_format: Optional[str],
    outfile_name: str = "config.json",
    verbose: bool = False,
) -> None:
    """
    Orchestrate interactive experiment setup with user prompts.
    This function handles all user interaction for experiment setup.
    """
    configs = ExperimentConfig.parse_configs(
        input_path=input_path,
        sheet=sheet,
        force_format=force_format,
    )
    if not configs:
        print("No experiments to process.")
        return

    # Prepare experiment directories (same logic as experiment command)
    experiments = prepare_experiment_directories(configs)
    if not experiments:
        print("No experiments to process.")
        return
    
    # Determine if using shared or individual ingredients
    use_shared_ingredients = True
    if len(experiments) > 1:
        use_shared_ingredients = prompt_yes_no("Use the same ingredients file for all experiments? (Y/N): ")
    
    # Get ingredients data
    shared_ingredients = None
    individual_ingredients: Dict[Path, Dict[str, Any]] = {}
    
    if use_shared_ingredients:
        shared_ingredients = prompt_for_ingredients()
    else:
        # Get individual ingredients for each experiment
        for cfg, exp_dir in experiments:
            print(f"\nExperiment directory: {exp_dir}")
            individual_ingredients[exp_dir] = prompt_for_ingredients()

    # Write files using pure functions
    config_p, config_verbose, ingredients_p, ingredients_verbose = write_experiment_files(
        experiments=experiments,
        shared_ingredients=shared_ingredients,
        individual_ingredients=individual_ingredients,
        outfile_name=outfile_name,
    )
    if verbose:
        print(config_verbose)
        print(ingredients_verbose)
    else:
        print(config_p)
        print(ingredients_p)


def run(
    experiment_path: str,
    plot_only: bool = False,
    verbose: bool = False,
) -> None:
    """
    Execute an experiment round using BacterAI models and simulations.
    The round number is automatically determined based on existing Round folders.
    """
    
    # Check if the experiment directory exists
    if not os.path.exists(experiment_path):
        print(f"Error: Experiment directory not found: {experiment_path}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(experiment_path):
        print(f"Error: Path is not a directory: {experiment_path}", file=sys.stderr)
        sys.exit(1)
    
    # Import the core run functionality
    try:
        
        # Get the round number automatically
        next_round = get_round(experiment_path)
        
        if verbose:
            print(f"Starting BacterAI run for round {next_round}")
            print(f"Experiment directory: {experiment_path}")
            print(f"Plot only: {plot_only}")
        
        # Execute the experiment
        execute_experiment(experiment_path, plot_only)
        
        if verbose:
            print(f"Run completed successfully for round {next_round}")
            
    except Exception as e:
        print(f"Error during experiment execution: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)