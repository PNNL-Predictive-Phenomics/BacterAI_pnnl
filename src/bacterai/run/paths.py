"""Path utilities for BacterAI experiments."""

from pathlib import Path

def get_round(experiment_path: str) -> int:
    """
    Automatically determine the next round number based on existing Round folders.
    """
    path = Path(experiment_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {experiment_path}")
    
    # Extract round numbers from Round directories
    round_nums = [
        int(item.name[5:]) 
        for item in path.iterdir() 
        if item.is_dir() and item.name.startswith("Round") and item.name[5:].isdigit()
    ]
    
    return max(round_nums) + 1 if round_nums else 1

def get_round_folder_path(base_path: str, round_num: int) -> Path:
    return Path(base_path) / f"Round{round_num}"

def setup_round_folders(experiment_path: str, round_number: int):
    """
    Setup paths for previous, current, and new round folders.
    
    Returns
    -------
    tuple
        (prev_round_folder, current_round_folder, new_round_folder)
    """
    prev_round_folder = get_round_folder_path(experiment_path, round_number - 2)
    current_round_folder = get_round_folder_path(experiment_path, round_number - 1)
    new_round_folder = get_round_folder_path(experiment_path, round_number)
    
    return prev_round_folder, current_round_folder, new_round_folder