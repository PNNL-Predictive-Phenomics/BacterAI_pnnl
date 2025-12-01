"""Data preparation utilities for BacterAI experiments."""

import os
import pandas as pd
import numpy as np
from ..sim.core import SimType


def generate_random_training_data(ingredients_pd, ingredients_map, settings, new_round_folder):
    """
    Generate random training data for cold start (Round 1 without transfer learning).
    
    Parameters
    ----------
    ingredients_pd : pd.DataFrame
        Ingredients dataframe
    ingredients_map : dict
        Mapping of ingredient indices to names  
    settings : Settings
        Experiment settings
    new_round_folder : Path
        Path to new round folder
        
    Returns
    -------
    tuple
        (X_train, y_train, used_experiments, redo_experiments)
    """
    print("Generating random training data for cold start...")
    
    n_ingredients = len(ingredients_map)
    n_examples = 1000
    
    # Create random inputs between MIN_VALUE and MAX_VALUE
    min_values = ingredients_pd["MIN_VALUE"].to_numpy()
    max_values = ingredients_pd["MAX_VALUE"].to_numpy()
    X_train = min_values + np.random.rand(n_examples, n_ingredients) * (max_values - min_values)
    
    # Round discrete types
    discrete_types = ingredients_pd["TYPE"].isin(["binary", "semi-quantitative"])
    X_train[:, discrete_types] = np.round(X_train[:, discrete_types])
    
    # Generate random fitness values [0, 1]
    y_train = np.random.rand(n_examples, 1).flatten()
    
    # Force at least 25% of the fitnesses to 0
    index_choices = set(range(len(y_train)))
    y_train_zeros = np.random.choice(
        list(index_choices), size=int(n_examples * 0.25), replace=False
    )
    y_train[y_train_zeros] = 0
    
    # Force at least 25% of the fitnesses to 1 (does not overwrite the zeroed out ones)
    index_choices -= set(y_train_zeros)
    y_train_ones = np.random.choice(
        list(index_choices), size=int(n_examples * 0.25), replace=False
    )
    y_train[y_train_ones] = 1
    
    # Update simulation types to RANDOM for this case
    settings.simulation_types = [SimType.RANDOM]
    
    # Create and save random training data
    data = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
    col_names = ingredients_map.copy()
    col_names[n_ingredients] = "y_train"
    data = data.rename(columns=col_names)
    
    # Save the random kickstart data
    random_data_filename = f"random_train_kickstart_{'aas' if settings.aas_only else 'others'}.csv"
    data.to_csv(os.path.join(new_round_folder, random_data_filename), index=False)
    
    print(f"Generated {n_examples} random training examples")
    
    return X_train, y_train, None, None  # used_experiments=None, redo_experiments=None