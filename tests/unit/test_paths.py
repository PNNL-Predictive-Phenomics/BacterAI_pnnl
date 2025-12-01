"""Unit tests for path utilities."""
import pytest
from pathlib import Path
from src.bacterai.run.paths import (
    get_round,
    get_round_folder_path,
    setup_round_folders,
)


class TestPaths:
    """Test suite for paths module."""

    def test_get_round_folder_path(self):
        """Test round folder path generation."""
        base_path = "/test/experiment"
        round_num = 3
        expected = Path("/test/experiment/Round3")
        result = get_round_folder_path(base_path, round_num)
        assert result == expected

    def test_get_round_nonexistent_directory(self):
        """Test get_round with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            get_round("/nonexistent/directory")

    def test_get_round_with_no_rounds(self, temp_experiment_dir):
        """Test get_round returns 1 when no Round folders exist."""
        result = get_round(str(temp_experiment_dir))
        assert result == 1

    def test_get_round_with_existing_rounds(self, temp_experiment_dir):
        """Test get_round finds the next round number."""
        # Create some Round folders
        (temp_experiment_dir / "Round1").mkdir()
        (temp_experiment_dir / "Round2").mkdir()
        (temp_experiment_dir / "Round3").mkdir()
        
        result = get_round(str(temp_experiment_dir))
        assert result == 4

    def test_get_round_with_non_sequential_rounds(self, temp_experiment_dir):
        """Test get_round with non-sequential round numbers."""
        (temp_experiment_dir / "Round1").mkdir()
        (temp_experiment_dir / "Round5").mkdir()
        (temp_experiment_dir / "Round10").mkdir()
        
        result = get_round(str(temp_experiment_dir))
        assert result == 11  # Max is 10, so next is 11

    def test_setup_round_folders(self):
        """Test setup_round_folders returns correct paths."""
        experiment_path = "/test/experiment"
        round_number = 3
        
        prev, current, new = setup_round_folders(experiment_path, round_number)
        
        assert prev == Path("/test/experiment/Round1")
        assert current == Path("/test/experiment/Round2")
        assert new == Path("/test/experiment/Round3")

    def test_setup_round_folders_first_round(self):
        """Test setup_round_folders for first round (Round1)."""
        experiment_path = "/test/experiment"
        round_number = 1
        
        prev, current, new = setup_round_folders(experiment_path, round_number)
        
        # Round -1 and Round 0 don't make sense, but function handles it
        assert prev == Path("/test/experiment/Round-1")
        assert current == Path("/test/experiment/Round0")
        assert new == Path("/test/experiment/Round1")
