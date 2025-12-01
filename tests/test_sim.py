import unittest
import os
import pickle
import numpy as np
import pytest
from src.bacterai.run.models import GPRModel, NeuralNetModel, ModelType
from src.bacterai.sim.core import rollout_trajectory, perform_simulations
from src.bacterai.sim.core import SimType, SimDirection

# Path to test data
test_data_path = os.path.join(os.path.dirname(__file__), 'test_experiment', 'test_data.pkl')

class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test data for the class
        with open(test_data_path, 'rb') as f:
            cls.test_data = pickle.load(f)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_rollout_trajectory(self):
        # Read arguments from the test data
        model, starting_media, ingredients_pd, new_round_n, batch_size, \
            sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
            go_beyond_frontier, used_experiments, redo_experiments = self.test_data
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
        # Asset the output shape
        self.assertEqual(rollout_results.shape, (3, ))

    def test_perform_simulations(self):
        # Read arguments from the test data
        model, starting_media, ingredients_pd, new_round_n, batch_size, \
            sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
            go_beyond_frontier, used_experiments, redo_experiments = self.test_data
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
        # Asset the output shape
        self.assertEqual(batch.shape, (100, 12))
