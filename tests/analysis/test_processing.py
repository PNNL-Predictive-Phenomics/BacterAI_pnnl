"""Tests for analysis processing and plotting utilities."""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from src.bacterai.analysis.processing import (
    process_mapped_data,
    normalize_ingredient_names,
    softmax,
    combined_round_data,
)
from src.bacterai.analysis.plotting import plot_results, plot_redos


def test_process_mapped_data(sample_mapped_data_df, temp_experiment_dir):
    data_path = temp_experiment_dir / "mapped_data.csv"
    sample_mapped_data_df.to_csv(data_path, index=False)

    processed, controls, blanks = process_mapped_data(
        str(data_path),
        ingredients=["glucose", "amino_acid_1", "vitamin_b"],
    )

    assert len(controls) == 1
    assert len(blanks) == 1
    assert "fitness" in processed.columns
    assert len(processed) == 2
    assert processed["fitness"].iloc[0] == 0.5


def test_normalize_ingredient_names_maps_constants():
    df = pd.DataFrame({"dl_alanine_75x": [1], "feature": [0.1]})
    out = normalize_ingredient_names(df)
    assert "ala" in out.columns


def test_softmax_handles_overflow():
    scores = np.array([1000.0, 1000.0])
    out = softmax(scores, k=1)
    assert np.isclose(out.sum(), 1.0)
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_combined_round_data_filters_redos(sample_results_all_df, temp_experiment_dir):
    r1 = temp_experiment_dir / "Round1"
    r1.mkdir()
    r2 = temp_experiment_dir / "Round2"
    r2.mkdir()

    sample_results_all_df.to_csv(r1 / "results_all_round1.csv", index=False)
    sample_results_all_df.to_csv(r2 / "results_all_round2.csv", index=False)

    combined = combined_round_data(str(temp_experiment_dir), sort=True)
    assert "is_redo" in combined.columns
    assert combined["is_redo"].sum() == 0

    block_size = sample_results_all_df[sample_results_all_df["is_redo"] == False].shape[0]
    first_block = combined.iloc[:block_size]["growth_pred"].tolist()
    second_block = combined.iloc[block_size:]["growth_pred"].tolist()
    assert first_block == sorted(first_block)
    assert second_block == sorted(second_block)


def test_plot_results_and_redos_creates_files(sample_results_all_df, temp_experiment_dir):
    plot_results(str(temp_experiment_dir), sample_results_all_df, threshold=0.25)

    assert os.path.exists(temp_experiment_dir / "results_graphic.png")

    ingredients = ["glucose", "amino_acid_1", "vitamin_b"]
    prev = sample_results_all_df.copy()
    redo = sample_results_all_df.copy()
    for df in (prev, redo):
        df["fitness"] = df["fitness"].astype(float)

    plot_redos(str(temp_experiment_dir), prev, redo, ingredients)
    assert os.path.exists(temp_experiment_dir / "redo_compare_order_plot.png")
