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
