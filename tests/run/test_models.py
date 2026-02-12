"""Tests for model wrappers."""
import numpy as np
import pytest

from src.bacterai.run.models import GPRModel, NeuralNetModel


def test_gpr_evaluate_requires_training(temp_experiment_dir):
    model = GPRModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))


def test_neural_net_evaluate_requires_training(temp_experiment_dir):
    model = NeuralNetModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))
