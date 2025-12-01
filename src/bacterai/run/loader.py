"""Configuration and ingredients loading utilities."""

import json
import os
import pandas as pd
from typing import Dict, Any


def load_experiment_config(experiment_path: str) -> Dict[str, Any]:
    """Load experiment configuration from config.json."""
    config_path = os.path.join(experiment_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path) as f:
        return json.load(f)


def load_ingredients_data(experiment_path: str, config: Dict[str, Any]):
    """Load and process ingredients data."""
    ingredients_file = config.get("ingredients_file", None)
    if ingredients_file is None:
        raise ValueError("No ingredients_file specified in config")
        
    ingredients_full_path = os.path.join(experiment_path, ingredients_file)
    with open(ingredients_full_path, "r") as f:
        ingredients_json = json.load(f)

    ingredients_pd = pd.json_normalize(ingredients_json["ingredients"])
    
    # Remove entries in the ingredients not relevant to the BacterAI logical loop
    # 1) remove grouped reagents which together create a single condition
    ingredients_pd = ingredients_pd[~(ingredients_pd.INGREDIENT.str.contains("\\:\\:"))]
    # 2) remove any other "ingredient" that is invariant -- where the MIN_VALUE and MAX_VALUE are the same
    ingredients_pd = ingredients_pd[~(ingredients_pd["MIN_VALUE"] == ingredients_pd["MAX_VALUE"])]
    # re-index
    ingredients_pd = ingredients_pd.reset_index(drop=True)
    
    return ingredients_pd