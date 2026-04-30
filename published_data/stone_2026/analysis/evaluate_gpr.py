"""Evaluate a trained GPR model from Round N on that round's results_all.csv (test data).
This script was created by Github Copilot Claude Opus 4.6 and edited by hand.

Usage:
    python evaluate_gpr.py <experiment_path> <round_number> [--threshold 0.25]

Example:
    python evaluate_gpr.py published_data/stone_2026/expt_rounds 2 --threshold 0.25

The Round N model was trained on Round N-1 data, so Round N's results_all.csv
is unseen test data for that model.
"""

import argparse
import os
import sys

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from gpr import ExactGPModel


def load_gpr_model(model_dir, n_features):
    """Load a trained GPR model from state dicts.

    The model was saved via torch.save(model.state_dict(), ...), so we
    reconstruct the model architecture with dummy data, then load weights.
    """
    model_path = os.path.join(model_dir, "gpr_model.pth")
    likelihood_path = os.path.join(model_dir, "gpr_likelihood.pth")

    if not os.path.exists(model_path) or not os.path.exists(likelihood_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}")

    # Create dummy training data to initialize the model structure
    dummy_x = torch.zeros(1, n_features, dtype=torch.float32)
    dummy_y = torch.zeros(1, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(dummy_x, dummy_y, likelihood)

    # Load saved state dicts
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    likelihood.load_state_dict(torch.load(likelihood_path, map_location="cpu", weights_only=True))

    model.eval()
    likelihood.eval()

    return model, likelihood


def predict(model, likelihood, X):
    """Get mean predictions from the GPR model."""
    test_x = torch.tensor(X, dtype=torch.float32).contiguous()

    # Update model's training data shape so prediction works
    # ExactGP needs to know the training inputs for posterior computation;
    # the state_dict stores kernel hyperparams but not the training data.
    # We must reload the actual training data — but since we only have
    # the state dict, we use the model in "fantasy model" mode or
    # reattach training data. For ExactGP, the training data is stored
    # in the model object and was set to dummy data above.
    # We need the actual training data to compute the posterior.

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
        output = model(test_x)
        mean = output.mean.numpy()

    return mean


def load_training_data(experiment_path, round_number):
    """Load the training data that was used to train the Round N model.

    For Round 2+, the training data is in train_pred.csv in the round folder
    (contains accumulated data from previous rounds with y_true, y_pred, y_pred_var).

    For Round 1, the model was trained on random kickstart data. That CSV only
    stores X (ingredient columns 0..N-1, 0-indexed) with no y column, so we
    cannot reconstruct the ExactGP posterior from it alone.
    """
    round_folder = os.path.join(experiment_path, f"Round{round_number}")

    train_pred_path = os.path.join(round_folder, "train_pred.csv")
    if os.path.exists(train_pred_path):
        data = pd.read_csv(train_pred_path)
        n_ingredients = len(data.columns) - 3  # subtract y_true, y_pred, y_pred_var
        X = data.iloc[:, :n_ingredients].to_numpy()
        y = data["y_true"].to_numpy()
        return X.astype(np.float32), y.astype(np.float32)

    # Fallback: Round 1 kickstart file (X only, no y)
    kickstart_files = [f for f in os.listdir(round_folder)
                       if f.startswith("random_train_kickstart")]
    if not kickstart_files:
        raise FileNotFoundError(
            f"No train_pred.csv or random_train_kickstart file in {round_folder}"
        )
    raise ValueError(
        f"Round {round_number} only has a kickstart file (no y_train column). "
        f"ExactGP requires training targets to compute posterior predictions. "
        f"Evaluation is only supported for rounds with a train_pred.csv (round 2+)."
    )


def load_test_data(experiment_path, round_number, n_ingredients):
    """Load results_all.csv from the given round as test data.

    Uses the first n_ingredients columns (by position) as features,
    matching the training data format.
    """
    round_folder = os.path.join(experiment_path, f"Round{round_number}")
    results_path = os.path.join(round_folder, "results_all.csv")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results_all.csv not found in {round_folder}")

    results = pd.read_csv(results_path)

    # Filter out bad wells
    if "bad" in results.columns:
        results = results[results["bad"] == 0]

    ingredient_columns = list(results.columns[:n_ingredients])
    X_test = results[ingredient_columns].to_numpy().astype(np.float32)
    y_test = results["fitness"].to_numpy().astype(np.float32)

    return X_test, y_test


def evaluate(y_true, y_pred, threshold):
    """Compute regression and classification metrics."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Classification: growth above threshold
    y_true_class = (y_true >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)

    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    accuracy = np.mean(y_true_class == y_pred_class)

    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1]).ravel()

    return {
        "mse": mse,
        "r2": r2,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_samples": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GPR model on held-out round data."
    )
    parser.add_argument("experiment_path", help="Path to experiment folder containing Round<N> dirs")
    parser.add_argument("round_number", type=int, help="Round number to evaluate (model from this round, tested on this round's results)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Growth threshold for classification metrics (default: 0.25)")
    args = parser.parse_args()

    model_dir = os.path.join(args.experiment_path, f"Round{args.round_number}", "gpr_model")

    # Load the training data first to determine n_ingredients
    # (needed by ExactGP to compute posterior predictions)
    print(f"Loading training data for Round{args.round_number} model...")
    X_train, y_train = load_training_data(args.experiment_path, args.round_number)
    n_ingredients = X_train.shape[1]
    print(f"  Training data: {X_train.shape[0]} samples, {n_ingredients} features")

    print(f"Loading GPR model from Round{args.round_number}...")
    model, likelihood = load_gpr_model(model_dir, n_ingredients)

    # Re-attach training data to the model for posterior computation
    train_x = torch.tensor(X_train, dtype=torch.float32).contiguous()
    train_y = torch.tensor(y_train, dtype=torch.float32).contiguous()
    model.set_train_data(train_x, train_y, strict=False)

    # Load test data
    print(f"Loading test data from Round{args.round_number}/results_all.csv...")
    X_test, y_test = load_test_data(args.experiment_path, args.round_number, n_ingredients)
    print(f"  Test data: {X_test.shape[0]} samples")

    # Predict
    print("Running predictions...")
    y_pred = predict(model, likelihood, X_test)

    # Evaluate
    metrics = evaluate(y_test, y_pred, args.threshold)

    # Print results
    print(f"\n{'='*50}")
    print(f"GPR Model Evaluation — Round {args.round_number}")
    print(f"{'='*50}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size:  {metrics['n_samples']}")
    print()
    print("Regression Metrics (test set):")
    print(f"  MSE:        {metrics['mse']:.4f}")
    print(f"  R2:         {metrics['r2']:.4f}")
    print()
    print(f"Classification Metrics (threshold = {args.threshold}):")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']:>5}  |  FP: {metrics['fp']:>5}")
    print(f"  FN: {metrics['fn']:>5}  |  TN: {metrics['tn']:>5}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
