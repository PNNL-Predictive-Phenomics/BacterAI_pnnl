"""Tests for model wrappers."""
import os
import numpy as np
import pytest
from types import SimpleNamespace

from src.bacterai.run.models import (
    GPRModel,
    NeuralNetModel,
    IterativeTransferRFModel,
    TimedTransferRFModel,
    ModelType,
)
from src.bacterai.run.model_training import train_experiment_model


def test_gpr_evaluate_requires_training(temp_experiment_dir):
    model = GPRModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))


def test_neural_net_evaluate_requires_training(temp_experiment_dir):
    model = NeuralNetModel(str(temp_experiment_dir))
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))


def test_iterative_transfer_rf_evaluate_requires_training():
    model = IterativeTransferRFModel()
    with pytest.raises(Exception):
        model.evaluate(np.zeros((2, 2)))


def test_iterative_transfer_rf_train_and_evaluate(mock_training_data):
    X, y = mock_training_data
    model = IterativeTransferRFModel(feature_names=["f0", "f1", "f2"])
    model.train(X, y, n_estimators=64, random_state=42)

    preds, variances = model.evaluate(X[:10])
    assert preds.shape == (10,)
    assert variances.shape == (10,)
    assert np.all(variances >= 0)


def test_iterative_transfer_rf_save_and_load(temp_experiment_dir, mock_training_data):
    X, y = mock_training_data
    model = IterativeTransferRFModel(feature_names=["f0", "f1", "f2"])
    model.train(X, y, n_estimators=64, random_state=42)

    artifact_path = temp_experiment_dir / "transfer_rf_model.pkl"
    model.save_trained_model(str(artifact_path))
    assert artifact_path.exists()

    loaded = IterativeTransferRFModel.load_trained_model(str(artifact_path))
    preds, variances = loaded.evaluate(X[:10])
    assert preds.shape == (10,)
    assert variances.shape == (10,)


def test_model_training_dispatches_iterative_transfer_rf(temp_experiment_dir, mock_training_data):
    X, y = mock_training_data
    settings = SimpleNamespace(
        transfer_learning=False,
        round_number=2,
        n_bags=25,
        model_type=ModelType.TRANSFER_RF,
        transfer_model_folder=None,
    )

    model = train_experiment_model(
        X,
        y,
        settings,
        str(temp_experiment_dir),
        transfer_model=None,
    )

    assert isinstance(model, IterativeTransferRFModel)
    preds, variances = model.evaluate(X[:5])
    assert preds.shape == (5,)
    assert variances.shape == (5,)

    artifact_path = os.path.join(str(temp_experiment_dir), "transfer_rf_model.pkl")
    assert os.path.exists(artifact_path)


class _DummyRF:
    def predict(self, X):
        n = len(X)
        return np.full(n, 0.42)


def test_timed_transfer_rf_save_and_load(temp_experiment_dir):
    artifact_path = temp_experiment_dir / "transfer_timed_hpc_rf_model.pkl"
    model = TimedTransferRFModel(classifier=_DummyRF(), feature_names=["f0", "f1"])
    model.save_trained_model(str(artifact_path))

    loaded = TimedTransferRFModel.load_trained_model(str(artifact_path))
    preds, variances = loaded.evaluate(np.zeros((4, 2)))
    assert preds.shape == (4,)
    assert variances.shape == (4,)
    assert np.allclose(preds, 0.42)


def test_model_training_warmstarts_from_timed_transfer_model(temp_experiment_dir, mock_training_data):
    X, y = mock_training_data
    settings = SimpleNamespace(
        transfer_learning=True,
        round_number=2,
        n_bags=25,
        model_type=ModelType.TRANSFER_RF,
        transfer_model_folder=None,
    )
    timed_model = TimedTransferRFModel(classifier=_DummyRF(), feature_names=["f0", "f1", "f2"])

    model = train_experiment_model(
        X,
        y,
        settings,
        str(temp_experiment_dir),
        transfer_model=timed_model,
    )

    assert isinstance(model, IterativeTransferRFModel)
    preds, variances = model.evaluate(np.zeros((3, 3)))
    assert preds.shape == (3,)
    assert variances.shape == (3,)
    assert np.allclose(preds, 0.42)
