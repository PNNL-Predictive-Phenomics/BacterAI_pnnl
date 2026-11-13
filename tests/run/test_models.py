"""Tests for model wrappers."""
import numpy as np
import pytest

from src.bacterai.run.models import GPRModel, NeuralNetModel, RandomForestModel


def test_gpr_evaluate_requires_training(temp_experiment_dir):
    model = GPRModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))

    # TODO: Add test_gpr_train_basic and test_gpr_load_trained_models


def test_neural_net_evaluate_requires_training(temp_experiment_dir):
    model = NeuralNetModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))

    # TODO: Add test_neural_net_train_basic and test_neural_net_load_trained_models


def test_random_forest_evaluate_requires_training(temp_experiment_dir):
    """Test that evaluate raises exception before training."""
    model = RandomForestModel(str(temp_experiment_dir))
    with pytest.raises(RuntimeError, match="needs to be trained"):
        model.evaluate(np.zeros((2, 2)))
