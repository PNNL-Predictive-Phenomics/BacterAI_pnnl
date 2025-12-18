"""
Pytest configuration and shared fixtures for BacterAI tests.
"""
import os
import pytest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_experiment():
    """
    Session-level fixture that preserves and restores the original mapped_data CSV.
    Saves a backup before tests run, then restores it and cleans up Round folders after tests complete.
    """
    exp_path = os.path.join(os.path.dirname(__file__), 'test_experiment')
    round_test_folder = os.path.join(exp_path, "Round_test")
    mapped_data_filename = "mapped_data_test_date_biotek_delta_od_data.csv"
    backup_filename = "mapped_data_test_date_biotek_delta_od_data (1).csv"
    
    original_file = os.path.join(round_test_folder, mapped_data_filename)
    backup_file = os.path.join(round_test_folder, backup_filename)
    temp_backup_file = os.path.join(round_test_folder, ".mapped_data_backup.csv")
    
    # Save backup before tests run (use the (1).csv as the canonical original)
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, temp_backup_file)
    elif os.path.exists(original_file):
        shutil.copy2(original_file, temp_backup_file)
    
    yield  # Tests run here
    
    # After all tests complete, restore from backup
    if os.path.exists(temp_backup_file):
        shutil.copy2(temp_backup_file, original_file)
        os.remove(temp_backup_file)
        print(f"\nRestored original {mapped_data_filename} to Round_test")
    
    # Clean up Round folders
    round_1_folder = os.path.join(exp_path, "Round1")
    round_2_folder = os.path.join(exp_path, "Round2")
    
    for folder in [round_1_folder, round_2_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleaned up {os.path.basename(folder)}")
    
    # Clean up any other Round folders except Round_test
    for item in os.listdir(exp_path):
        if item.startswith("Round") and item != "Round_test":
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Cleaned up {item}")


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary experiment directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "grow_threshold": 0.25,
        "experiment_path": "tests/test_experiment",
        "nickname": "test_exp",
        "batch_size": 96,
        "timeout_min": 15,
        "n_rollouts": 10,
        "model_type": 1,  # NEURAL_NET
        "direction": 0,  # DOWN
        "simulation_types": [2],  # ROLLOUT
        "beyond_frontier": True,
        "use_unique": True,
        "ingredients_file": "ingredients.json",
        "redo_size": 0,
        "redo_threshold": [0, 1],
        "aas_only": False,
        "separate_redos": False,
        "n_bags": 25,
        "random_walk_increment": 10,
    }


@pytest.fixture
def sample_ingredients_data():
    """Sample ingredients dataframe."""
    return pd.DataFrame({
        "INGREDIENT": ["glucose", "amino_acid_1", "vitamin_b"],
        "TYPE": ["quantitative", "binary", "semi-quantitative"],
        "MIN_VALUE": [0.0, 0.0, 0.0],
        "MAX_VALUE": [10.0, 1.0, 5.0],
        "NOMINAL_VALUE": [5.0, 1.0, 2.5],
        "N_STATES": [10, 2, 3],
    })


@pytest.fixture
def sample_ingredients_json(sample_ingredients_data):
    """Sample ingredients JSON structure."""
    ingredients_list = []
    for _, row in sample_ingredients_data.iterrows():
        ingredients_list.append({
            "INGREDIENT": row["INGREDIENT"],
            "TYPE": row["TYPE"],
            "MIN_VALUE": row["MIN_VALUE"],
            "MAX_VALUE": row["MAX_VALUE"],
            "NOMINAL_VALUE": row["NOMINAL_VALUE"],
            "N_STATES": int(row["N_STATES"]),
        })
    return {"ingredients": ingredients_list}


@pytest.fixture
def mock_training_data():
    """Generate mock training data."""
    n_samples = 50
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    return X, y


@pytest.fixture
def sample_batch_df():
    """Sample batch dataframe with experiment results."""
    return pd.DataFrame({
        0: [1, 1, 0, 1, 0],
        1: [1, 0, 1, 1, 0],
        2: [1, 1, 1, 0, 1],
        "type": ["ROLLOUT", "RANDOM", "GREEDY", "ROLLOUT", "RANDOM"],
        "direction": ["DOWN", "DOWN", "UP", "DOWN", "UP"],
        "frontier_type": ["FRONTIER", "BEYOND", "FRONTIER", "FRONTIER", "BEYOND"],
        "growth_pred": [0.8, 0.3, 0.9, 0.7, 0.2],
        "var": [0.1, 0.2, 0.05, 0.15, 0.25],
        "is_redo": [False, False, False, False, False],
        "round": [1, 1, 1, 1, 1],
    })


@pytest.fixture
def setup_experiment_dir(temp_experiment_dir, sample_config, sample_ingredients_json):
    """Setup a complete experiment directory structure."""
    # Create config.json
    config_path = temp_experiment_dir / "config.json"
    sample_config["experiment_path"] = str(temp_experiment_dir)
    with open(config_path, "w") as f:
        json.dump(sample_config, f, indent=4)
    
    # Create ingredients.json
    ingredients_path = temp_experiment_dir / "ingredients.json"
    with open(ingredients_path, "w") as f:
        json.dump(sample_ingredients_json, f, indent=4)
    
    # Create Round1 directory with mock data
    round1_dir = temp_experiment_dir / "Round1"
    round1_dir.mkdir()
    
    # Create mock mapped_data CSV
    mapped_data = pd.DataFrame({
        "experiment_number": [1, 2, 3],
        "glucose": [1, 0, 1],
        "amino_acid_1": [1, 1, 0],
        "vitamin_b": [1, 1, 1],
        "feature": [0.8, 0.3, 0.9],
        "plate_control": [False, False, False],
        "plate_blank": [False, False, False],
        "parent_plate": ["plate1", "plate1", "plate1"],
        "bad": [0, 0, 0],
    })
    mapped_data.to_csv(round1_dir / "mapped_data.csv", index=False)
    
    # Create mock batch_meta CSV
    batch_meta = pd.DataFrame({
        "glucose": [1, 0, 1],
        "amino_acid_1": [1, 1, 0],
        "vitamin_b": [1, 1, 1],
        "growth_pred": [0.8, 0.3, 0.9],
        "var": [0.1, 0.2, 0.15],
        "type": ["ROLLOUT", "RANDOM", "GREEDY"],
        "is_redo": [False, False, False],
    })
    batch_meta.to_csv(round1_dir / "batch_meta.csv", index=False)
    
    return temp_experiment_dir


@pytest.fixture(scope="session")
def simulation_test_data():
    """
    Generate test data for simulation and batch tests.
    This replaces the old pickle file with fresh data generated from test_experiment.
    
    Returns a tuple of 14 items:
    (model, starting_media, ingredients_pd, new_round_n, batch_size,
     sim_types, rollout_trajectories, threshold, timeout, unique, direction,
     go_beyond_frontier, used_experiments, redo_experiments)
    """
    from src.bacterai.run.models import GPRModel
    from src.bacterai.sim.core import SimType, SimDirection
    
    # Paths to test experiment data
    test_exp_path = os.path.join(os.path.dirname(__file__), 'test_experiment')
    model_path = os.path.join(test_exp_path, 'Round_test', 'gpr_model')
    config_path = os.path.join(test_exp_path, 'config.json')
    ingredients_path = os.path.join(test_exp_path, 'ingredients.json')
    results_path = os.path.join(test_exp_path, 'Round_test', 'results_grow_only.csv')
    
    # Load the trained model
    model = GPRModel.load_trained_models(model_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load ingredients and create DataFrame
    with open(ingredients_path, 'r') as f:
        ingredients_json = json.load(f)
    
    # Filter to only quantitative/binary/semi-quantitative (not environment or fill)
    ingredients_list = [
        ing for ing in ingredients_json['ingredients']
        if ing['TYPE'] in ['quantitative', 'binary', 'semi-quantitative']
    ]
    
    # Convert N_STATES to int, defaulting to 10 for quantitative types with None
    for ing in ingredients_list:
        if ing['N_STATES'] is None:
            ing['N_STATES'] = 10  # Default for quantitative
        elif isinstance(ing['N_STATES'], str):
            ing['N_STATES'] = int(ing['N_STATES'])
    
    ingredients_pd = pd.DataFrame(ingredients_list)
    n_ingredients = len(ingredients_list)
    
    # Load results and get starting media from first row
    results_df = pd.read_csv(results_path)
    ingredient_names = [ing['INGREDIENT'] for ing in ingredients_list]
    starting_media = results_df.iloc[0][ingredient_names].values.astype(np.float64)
    
    # Create test parameters from config
    new_round_n = 2
    batch_size = config.get('batch_size', 100)
    sim_types = [SimType(st) for st in config.get('simulation_types', [2])]
    rollout_trajectories = config.get('n_rollouts', 2)
    threshold = config.get('grow_threshold', 0.25)
    timeout = config.get('timeout_min', 60) * 60  # Convert to seconds
    unique = config.get('use_unique', False)
    direction = SimDirection(config.get('direction', 0))
    go_beyond_frontier = config.get('beyond_frontier', True)
    used_experiments = set()
    redo_experiments = []
    
    return (
        model,
        starting_media,
        ingredients_pd,
        new_round_n,
        batch_size,
        sim_types,
        rollout_trajectories,
        threshold,
        timeout,
        unique,
        direction,
        go_beyond_frontier,
        used_experiments,
        redo_experiments,
    )
