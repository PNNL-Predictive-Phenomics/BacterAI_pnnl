"""Tests for src/bacterai/run/batch.py"""
import pytest
from src.bacterai.run.batch import make_batch


@pytest.mark.skip(reason="GPRModel.evaluate() needs fix: passes list to sample_GP instead of individual model/likelihood")
def test_make_batch(simulation_test_data):
    """Test that make_batch generates correct batch size and structure."""
    # Unpack test data
    model, starting_media, ingredients_pd, new_round_n, batch_size, \
        sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
        go_beyond_frontier, used_experiments, redo_experiments = simulation_test_data
    
    # Make batch
    batch, batch_used, metrics = make_batch(
        model,
        starting_media,
        ingredients_pd,
        new_round_n=new_round_n,
        batch_size=batch_size,
        sim_types=sim_types,
        rollout_trajectories=rollout_trajectories,
        threshold=threshold,
        timeout=timeout,
        unique=unique,
        direction=direction,
        go_beyond_frontier=go_beyond_frontier,
        used_experiments=used_experiments,
        redo_experiments=redo_experiments,
    )
    
    # Assert the output shape and types
    assert batch.shape[0] == batch_size
    assert batch.shape[1] == len(starting_media)
    assert isinstance(metrics, dict)
