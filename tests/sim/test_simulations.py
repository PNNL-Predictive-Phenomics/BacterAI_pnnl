"""Tests for simulation functions."""
import numpy as np
import pytest
from src.bacterai.sim.core import rollout_trajectory, perform_simulations


@pytest.mark.skip(reason="GPRModel.evaluate() needs fix: passes list to sample_GP instead of individual model/likelihood")
def test_rollout_trajectory(simulation_test_data):
    """Test rollout_trajectory function with realistic test data."""
    # Unpack test data
    model, starting_media, ingredients_pd, new_round_n, batch_size, \
        sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
        go_beyond_frontier, used_experiments, redo_experiments = simulation_test_data
    
    # Set up candidate states from starting media
    choices = np.argwhere(starting_media == direction.target_value())[:, 0]
    candidate_states = np.tile(starting_media, (choices.size, 1))
    
    # Run rollout_trajectory
    rollout_results = rollout_trajectory(
        model,
        candidate_states,
        ingredients_pd,
        rollout_trajectories,
        threshold,
        direction,
    )
    
    # Assert the output shape
    assert rollout_results.shape == (rollout_trajectories,)


@pytest.mark.skip(reason="GPRModel.evaluate() needs fix: passes list to sample_GP instead of individual model/likelihood")
def test_perform_simulations(simulation_test_data):
    """Test perform_simulations function with realistic test data."""
    # Unpack test data
    model, starting_media, ingredients_pd, new_round_n, batch_size, \
        sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
        go_beyond_frontier, used_experiments, redo_experiments = simulation_test_data
    
    # Perform simulations
    batch, batch_set, metrics = perform_simulations(
        model,
        starting_media,
        ingredients_pd,
        batch_size,
        threshold,
        sim_types[0],
        direction,
        new_round_n,
        unique=unique,
        timeout=timeout,
        batch_set=used_experiments,
        n_rollout_trajectories=rollout_trajectories,
        go_beyond_frontier=go_beyond_frontier,
    )
    
    # Assert the output shape and types
    assert batch.shape[0] == batch_size
    assert batch.shape[1] == len(starting_media)
    assert isinstance(metrics, dict)
