"""Tests for random forest utilities."""
import os
import numpy as np
import pytest
import tempfile
import shutil

from src.bacterai.ml.random_forest import train_new_RF, predict_RF, sample_RF, load_RF


def test_train_new_rf_basic():
    """Test basic training of Random Forest model."""
    temp_dir = tempfile.mkdtemp()
    try:
        X_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        model = train_new_RF(
            X_train,
            y_train,
            temp_dir,
            n_estimators=10,
            random_state=42,
        )

        assert model is not None
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 10
    finally:
        shutil.rmtree(temp_dir)


def test_train_new_rf_validation():
    """Test that train_new_rf validates input data."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Test NaN in X
        X_train = np.array([[np.nan, 2.0], [3.0, 4.0]])
        y_train = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="X contains NaN"):
            train_new_RF(X_train, y_train, temp_dir, n_estimators=5)

        # Test NaN in y
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_train = np.array([np.nan, 0.2])

        with pytest.raises(ValueError, match="y contains NaN"):
            train_new_RF(X_train, y_train, temp_dir, n_estimators=5)

        # Test shape mismatch
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_train = np.array([0.1])

        with pytest.raises(ValueError, match="same number of samples"):
            train_new_RF(X_train, y_train, temp_dir, n_estimators=5)
    finally:
        shutil.rmtree(temp_dir)


def test_predict_rf_returns_mean_and_variance():
    """Test that predict_RF returns mean predictions and variance."""
    temp_dir = tempfile.mkdtemp()
    try:
        X_train = np.random.rand(40, 4)
        y_train = np.random.rand(40)

        model = train_new_RF(X_train, y_train, temp_dir, n_estimators=10, random_state=42)

        X_test = np.random.rand(5, 4)
        mean_pred, variance = predict_RF(model, X_test)

        assert mean_pred.shape == (5,)
        assert variance.shape == (5,)
        assert np.all(variance >= 0)
    finally:
        shutil.rmtree(temp_dir)


def test_sample_rf_generates_samples():
    """Test that sample_RF generates approximate samples."""
    temp_dir = tempfile.mkdtemp()
    try:
        X_train = np.random.rand(40, 4)
        y_train = np.random.rand(40)

        model = train_new_RF(X_train, y_train, temp_dir, n_estimators=20, random_state=42)

        X_test = np.random.rand(3, 4)
        samples, variances = sample_RF(model, X_test, n_samples=50, random_state=42)

        assert samples.shape == (50, 3)
        assert variances.shape == (3,)
        assert np.all(variances >= 0)
    finally:
        shutil.rmtree(temp_dir)


def test_sample_rf_validation():
    """Test that sample_RF validates input."""
    temp_dir = tempfile.mkdtemp()
    try:
        X_train = np.random.rand(40, 4)
        y_train = np.random.rand(40)

        model = train_new_RF(X_train, y_train, temp_dir, n_estimators=10, random_state=42)

        # Test n_samples < 1
        X_test = np.random.rand(2, 4)
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            sample_RF(model, X_test, n_samples=0)
    finally:
        shutil.rmtree(temp_dir)


def test_load_rf_saves_and_loads():
    """Test that load_RF successfully loads a saved model."""
    temp_dir = tempfile.mkdtemp()
    try:
        X_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Train and save
        model1 = train_new_RF(X_train, y_train, temp_dir, n_estimators=10, random_state=42)

        # Load
        model2 = load_RF(temp_dir)

        # Verify they produce identical predictions
        X_test = np.random.rand(3, 5)
        pred1, var1 = predict_RF(model1, X_test)
        pred2, var2 = predict_RF(model2, X_test)

        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(var1, var2)
    finally:
        shutil.rmtree(temp_dir)


def test_load_rf_missing_file():
    """Test that load_RF raises error for missing model."""
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_RF("/nonexistent/path")
