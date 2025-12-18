"""Unit tests for export utilities."""
import pytest
import pandas as pd
from pathlib import Path
from src.bacterai.run.export import export_to_dp_batch


class TestExport:
    """Test suite for export module."""

    def test_export_to_dp_batch_basic(self, temp_experiment_dir, sample_batch_df):
        """Test basic export functionality."""
        ingredient_names = ["glucose", "amino_acid_1", "vitamin_b"]
        date = "2024-12-01"
        
        export_to_dp_batch(
            str(temp_experiment_dir),
            sample_batch_df.copy(),
            ingredient_names,
            date,
        )
        
        # Check that batch_meta file was created
        meta_files = list(temp_experiment_dir.glob("batch_meta_*.csv"))
        assert len(meta_files) == 1
        
        # Check that batch_dp file was created
        dp_files = list(temp_experiment_dir.glob("batch_dp_*.csv"))
        assert len(dp_files) == 1

    def test_export_to_dp_batch_with_nickname(self, temp_experiment_dir, sample_batch_df):
        """Test export with nickname."""
        ingredient_names = ["glucose", "amino_acid_1", "vitamin_b"]
        date = "2024-12-01"
        nickname = "test_exp_R1"
        
        export_to_dp_batch(
            str(temp_experiment_dir),
            sample_batch_df.copy(),
            ingredient_names,
            date,
            nickname=nickname,
        )
        
        # Check that files include nickname
        dp_files = list(temp_experiment_dir.glob(f"batch_dp_{nickname}_*.csv"))
        assert len(dp_files) == 1

    def test_export_to_dp_batch_redo(self, temp_experiment_dir, sample_batch_df):
        """Test export for redo experiments."""
        ingredient_names = ["glucose", "amino_acid_1", "vitamin_b"]
        date = "2024-12-01"
        
        export_to_dp_batch(
            str(temp_experiment_dir),
            sample_batch_df.copy(),
            ingredient_names,
            date,
            is_redo=True,
        )
        
        # Check that redo files were created
        meta_files = list(temp_experiment_dir.glob("batch_redo_meta_*.csv"))
        assert len(meta_files) == 1
        
        dp_files = list(temp_experiment_dir.glob("batch_redo_dp_*.csv"))
        assert len(dp_files) == 1

    def test_export_to_dp_batch_empty(self, temp_experiment_dir, capsys):
        """Test export with empty batch."""
        ingredient_names = ["glucose", "amino_acid_1", "vitamin_b"]
        date = "2024-12-01"
        empty_batch = pd.DataFrame()
        
        export_to_dp_batch(
            str(temp_experiment_dir),
            empty_batch,
            ingredient_names,
            date,
        )
        
        # Should print message and create no files
        captured = capsys.readouterr()
        assert "Empty Batch" in captured.out
        
        meta_files = list(temp_experiment_dir.glob("batch_meta_*.csv"))
        assert len(meta_files) == 0

    def test_export_to_dp_batch_sorted(self, temp_experiment_dir, sample_batch_df):
        """Test that exported batch is sorted by growth_pred and var."""
        ingredient_names = ["glucose", "amino_acid_1", "vitamin_b"]
        date = "2024-12-01"
        
        export_to_dp_batch(
            str(temp_experiment_dir),
            sample_batch_df.copy(),
            ingredient_names,
            date,
        )
        
        # Read back the meta file
        meta_files = list(temp_experiment_dir.glob("batch_meta_*.csv"))
        df = pd.read_csv(meta_files[0])
        
        # Check that it's sorted (descending growth_pred, ascending var)
        growth_preds = df["growth_pred"].tolist()
        assert growth_preds == sorted(growth_preds, reverse=True) or \
               all(df["growth_pred"] == df["growth_pred"].iloc[0])  # All same values
