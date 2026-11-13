"""Tests for model wrappers."""
import os
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


def test_random_forest_train_basic(temp_experiment_dir):
    """Test basic model training."""
    model = RandomForestModel(str(temp_experiment_dir))

    X_train = np.random.rand(50, 5)
    y_train = np.random.rand(50)

    trained_model = model.train(
        X_train,
        y_train,
        n_estimators=10,
        random_state=42,
    )

    assert model.is_trained
    assert trained_model is not None
    assert model.model is not None


def test_random_forest_load_trained_models(temp_experiment_dir):
    """Test loading a trained model."""
    model_path = str(temp_experiment_dir)
    model1 = RandomForestModel(model_path)

    X_train = np.random.rand(50, 5)
    y_train = np.random.rand(50)
    model1.train(X_train, y_train, n_estimators=10, random_state=42)

    # Load the trained model
    model2 = RandomForestModel.load_trained_models(model_path)

    assert model2.is_trained
    assert model2.model is not None

    # Verify that loaded model has the same underlying scikit-learn model
    X_test = np.random.rand(3, 5)

    from src.bacterai.ml.random_forest import predict_RF
    pred1_mean, var1 = predict_RF(model1.model, X_test)
    pred2_mean, var2 = predict_RF(model2.model, X_test)

    np.testing.assert_array_almost_equal(pred1_mean, pred2_mean)
    np.testing.assert_array_almost_equal(var1, var2)


def test_random_forest_load_trained_models_missing_file(temp_experiment_dir):
    """Test that loading from non-existent path raises error."""
    model_path = os.path.join(str(temp_experiment_dir), "nonexistent")

    with pytest.raises(FileNotFoundError, match="Random Forest model not found"):
        RandomForestModel.load_trained_models(model_path)
