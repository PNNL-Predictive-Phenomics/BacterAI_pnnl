"""Tests for data preparation utilities."""
import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.bacterai.run.data_preparation import generate_random_training_data
from src.bacterai.sim.core import SimType


def test_generate_random_training_data_shapes(temp_experiment_dir):
    ingredients_pd = pd.DataFrame({
        "INGREDIENT": ["glucose", "amino_acid_1", "vitamin_b"],
        "TYPE": ["quantitative", "binary", "semi-quantitative"],
        "MIN_VALUE": [0.0, 0.0, 0.0],
        "MAX_VALUE": [10.0, 1.0, 5.0],
    })
    ingredients_map = {0: "glucose", 1: "amino_acid_1", 2: "vitamin_b"}
    settings = SimpleNamespace(simulation_types=[SimType.ROLLOUT], aas_only=False)

    X_train, y_train, used, redo = generate_random_training_data(
        ingredients_pd,
        ingredients_map,
        settings,
        str(temp_experiment_dir),
    )

    assert X_train.shape[0] == 1000
    assert X_train.shape[1] == 3
    assert y_train.shape == (1000,)
    assert used is None
    assert redo is None
    assert settings.simulation_types == [SimType.RANDOM]

    binary_col = ingredients_pd["TYPE"].eq("binary").to_numpy()
    semi_col = ingredients_pd["TYPE"].eq("semi-quantitative").to_numpy()

    assert np.all(np.isin(X_train[:, binary_col], [0, 1]))
    assert np.all(np.mod(X_train[:, semi_col], 1) == 0)
