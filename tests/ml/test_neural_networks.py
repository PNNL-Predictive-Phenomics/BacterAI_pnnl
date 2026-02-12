"""Tests for neural network utilities."""
import numpy as np
import torch

from src.bacterai.ml.neural_networks import DatasetAminoAcids, threshold, accuracy, eval_bagged


def test_dataset_amino_acids_train_mode():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.1, 0.2])
    ds = DatasetAminoAcids(X, y, mode="train")

    assert len(ds) == 2
    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert isinstance(y0, torch.Tensor)


def test_threshold_and_accuracy():
    preds = np.array([0.1, 0.6, 0.3])
    labels = np.array([0.0, 1.0, 0.0])

    acc = accuracy(preds, labels, threshold=0.5)
    assert acc == 1.0


def test_eval_bagged_averages():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    class Dummy:
        def __init__(self, value):
            self.value = value

        def evaluate(self, _x):
            return np.array([self.value, self.value])

    models = [Dummy(0.2), Dummy(0.6)]
    pred, var = eval_bagged(X, models)

    assert np.allclose(pred, np.array([0.4, 0.4]))
    assert np.allclose(var, np.array([0.04, 0.04]))
