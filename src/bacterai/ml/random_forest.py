import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_new_RF(
    X,
    y,
    model_path,
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
):
    """
    Train a Random Forest regression model.

    Parameters
    ----------
    X : array-like
        Training features, shape (n_samples, n_features).

    y : array-like
        Training target values, shape (n_samples,).

    model_path : str
        Directory where the model will be saved.

    n_estimators : int
        Number of trees in the forest.

    max_depth : int or None
        Maximum depth of each tree.

    min_samples_split : int or float
        Minimum number of samples required to split a node.

    min_samples_leaf : int or float
        Minimum number of samples required at a leaf node.

    max_features : int, float, str, or None
        Number of features considered for each split.

    random_state : int
        Random seed.

    n_jobs : int
        Number of parallel jobs. Use -1 for all available cores.

    verbosity : int
        Controls the amount of output.

    Returns
    -------
    model : RandomForestRegressor
        Trained Random Forest model.
    """

    # Convert input data to NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Basic validation
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if np.isnan(X).any():
        raise ValueError("X contains NaN values. Impute or remove them first.")

    if np.isnan(y).any():
        raise ValueError("y contains NaN values. Remove them first.")

    # Initialize the Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbosity if verbosity > 1 else 0,
    )

    # Train the model
    model.fit(X, y)

    # Calculate predictions on the training data
    y_pred = model.predict(X)

    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)

    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R2: {final_r2:.4f}")

    # Create output directory
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    model_file = os.path.join(model_path, "random_forest_model.joblib")
    joblib.dump(model, model_file)

    print(f"Model saved to: {model_file}")

    return model


def sample_RF(model, X, n_samples=1, random_state=None):
    """
    Generate approximate Random Forest samples for input data.

    Each tree in a Random Forest produces one prediction. The predictions
    from the individual trees are treated as an empirical approximation
    to a predictive distribution.

    Parameters
    ----------
    model : RandomForestRegressor
        A trained Random Forest model.

    X : array-like
        Input features, shape (n_samples, n_features).

    n_samples : int
        Number of predictive samples to generate.

    random_state : int or None
        Random seed used when selecting trees.

    Returns
    -------
    samples : numpy.ndarray
        Approximate predictions, shape (n_samples, n_test_points).

    variances : numpy.ndarray
        Variance of the tree predictions for each test point,
        shape (n_test_points,).
    """

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if not hasattr(model, "estimators_"):
        raise ValueError("The model must be fitted before calling sample_RF().")

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    # Prediction from every tree
    tree_predictions = np.asarray([
        tree.predict(X)
        for tree in model.estimators_
    ])

    # Shape:
    # tree_predictions = (number_of_trees, number_of_test_points)

    # Estimate predictive variance from the individual trees
    if tree_predictions.shape[0] > 1:
        variances = np.var(tree_predictions, axis=0, ddof=1)
    else:
        variances = np.zeros(tree_predictions.shape[1])

    # Randomly select tree predictions to create samples
    rng = np.random.default_rng(random_state)

    selected_tree_indices = rng.choice(
        tree_predictions.shape[0],
        size=n_samples,
        replace=True,
    )

    samples = tree_predictions[selected_tree_indices]

    return samples, variances


def predict_RF(model, X):
    """
    Return the mean Random Forest prediction and tree-based variance.
    """

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    tree_predictions = np.asarray([
        tree.predict(X)
        for tree in model.estimators_
    ])

    mean_prediction = np.mean(tree_predictions, axis=0)

    if tree_predictions.shape[0] > 1:
        variance = np.var(tree_predictions, axis=0, ddof=1)
    else:
        variance = np.zeros(tree_predictions.shape[1])

    return mean_prediction, variance


def load_RF(model_path):
    """
    Load a saved Random Forest model.
    """

    model_file = os.path.join(model_path, "random_forest_model.joblib")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    return joblib.load(model_file)


# Usage
"""
import numpy as np

# Example training data
X_train = np.random.rand(100, 5)
y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(100)

# Train and save the model
model = train_new_RF(
    X=X_train,
    y=y_train,
    model_path="./models",
    n_estimators=300,
    max_depth=None,
    random_state=42,
)

# Test data
X_test = np.random.rand(10, 5)

# Mean prediction and estimated variance
mean_prediction, variances = predict_RF(model, X_test)

print("Mean predictions:")
print(mean_prediction)

print("Prediction variances:")
print(variances)

# Generate approximate predictive samples
samples, variances = sample_RF(
    model=model,
    X=X_test,
    n_samples=100,
    random_state=42,
)

print("Samples shape:", samples.shape)
print("Variances shape:", variances.shape)


"""
