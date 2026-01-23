"""
CLI command handlers for the unified configuration tool.
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
from bacterai.configuration.plate_readers import process_biotek_data, process_tecan_data
from bacterai.setup import write_experiment_files, scan_next_index
from bacterai.run.core import execute_experiment
from bacterai.run.paths import get_round
from .prompts import prompt_nonempty, prompt_yes_no, prompt_for_ingredients, handle_existing_directory


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


def process_data(
    reader_type: str,
    path: str,
    date: Optional[str],
    round_number: int,
    feature: str,
    signal: Optional[int] = None,
    output: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Process plate reader data (Biotek or Tecan) and create mapped_data CSV.
    
    This command extracts features from raw plate reader files and merges with
    experiment metadata to create the mapped_data file needed for BacterAI training.
    
    Args:
        reader_type: Type of plate reader ('biotek' or 'tecan')
        path: Path to experiment directory containing experiment_request/
        date: Date identifier for the data files (optional)
        round_number: Round number for the experiment
        feature: Feature to extract (e.g., 'delta_od', 'max_slope')
        signal: Wavelength signal for Biotek (required for Biotek, ignored for Tecan)
        output: Optional output filename override
        verbose: Print detailed processing information
    """
    
    # Validate inputs
    reader_type = reader_type.lower()
    if reader_type not in ['biotek', 'tecan']:
        print(f"Error: Invalid reader type '{reader_type}'. Must be 'biotek' or 'tecan'.", file=sys.stderr)
        sys.exit(1)
    
    # Convert path to absolute path
    exp_path = Path(path).expanduser().resolve()
    
    # Check if experiment directory exists
    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_path}", file=sys.stderr)
        sys.exit(1)
    
    if not exp_path.is_dir():
        print(f"Error: Path is not a directory: {exp_path}", file=sys.stderr)
        sys.exit(1)
    
    # Validate experiment_request directory structure - support multiple layouts
    exp_request_dir = exp_path / "experiment_request"
    round_exp_request_dir = exp_path / f"Round{round_number}" / "experiment_request"
    
    if not exp_request_dir.exists() and not round_exp_request_dir.exists():
        print(f"Error: experiment_request directory not found in {exp_path}", file=sys.stderr)
        print("Expected structure: <experiment_path>/experiment_request/ OR <experiment_path>/RoundN/experiment_request/", file=sys.stderr)
        sys.exit(1)
    
    # Check for config.json and ingredients.json
    config_file = exp_path / "config.json"
    ingredients_file = exp_path / "ingredients.json"
    
    if not config_file.exists():
        print(f"Warning: config.json not found in {exp_path}", file=sys.stderr)
    
    if not ingredients_file.exists():
        print(f"Warning: ingredients.json not found in {exp_path}", file=sys.stderr)
    
    # Biotek-specific validation
    if reader_type == 'biotek':
        if signal is None:
            print("Error: --signal is required for Biotek data processing.", file=sys.stderr)
            print("Example: bacterai process_data biotek /path/to/exp --date 2024-01-15 --round 1 --signal 600 --feature delta_od", file=sys.stderr)
            sys.exit(1)
        
        if not isinstance(signal, int) or signal <= 0:
            print(f"Error: Signal must be a positive integer (e.g., 600), got: {signal}", file=sys.stderr)
            sys.exit(1)
    
    # Tecan-specific info
    if reader_type == 'tecan' and signal is not None:
        if verbose:
            print("Note: --signal is ignored for Tecan data processing.")
    
    if verbose:
        print(f"Processing {reader_type.upper()} data:")
        print(f"  Experiment path: {exp_path}")
        print(f"  Date: {date if date else 'Not specified (using direct experiment_request path)'}")
        print(f"  Round: {round_number}")
        print(f"  Feature: {feature}")
        if reader_type == 'biotek':
            print(f"  Signal: {signal}")
        print()
    
    # Process the data
    try:
        if reader_type == 'biotek':
            result_df = process_biotek_data(
                path=str(exp_path),
                date=date,
                round_number=round_number,
                signal=signal,
                feature=feature,
            )
        else:  # tecan
            result_df = process_tecan_data(
                path=str(exp_path),
                date=date,
                round_number=round_number,
                feature=feature,
            )
        
        # Determine output filename
        if output:
            output_file = Path(output)
        else:
            # Use default naming from save_mapped_data
            round_dir = exp_path / f"Round{round_number}"
            date_str = date if date else sys.time.strftime("%Y%m%d")
            output_file = round_dir / f"mapped_data_{date_str}_{reader_type}_{feature}_data.csv"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        result_df.to_csv(output_file, index=False)

        # count unique experiments processed
        total_expt = result_df[result_df["experiment_number"] < 9999].shape[0]
        unique_expt = result_df[result_df["experiment_number"] < 9999]['experiment_number'].nunique()
        
        print(f"Successfully processed {total_expt} experiments ({unique_expt} unique experiments)")
        print(f"Output written to: {output_file}")
        
        if verbose:
            print(f"\nData preview (first 5 rows):")
            print(result_df.head())
            print(f"\nColumns: {', '.join(result_df.columns.tolist())}")
            print(f"Shape (incl. controls): {result_df.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid data format: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error during data processing: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        sys.exit(1)
