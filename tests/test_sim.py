import unittest
import os
import pickle
import numpy as np
from models import GPRModel, NeuralNetModel, ModelType
from sim import rollout_trajectory, perform_simulations
from sim import SimType, SimDirection

# Load test data for functions
test_data_path = os.path.join(os.path.dirname(__file__), 'test_experiment', 'test_data.pkl')
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the temporary directory and its contents
        #round_test_folder = os.path.join(EXPT_FOLDER, "Round_test")
        #round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
        #shutil.copytree(round_test_folder, round_folder)
        pass

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory and its contents
        #round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
        #shutil.rmtree(round_folder)
        pass

    def test_rollout_trajectory(self):
        # Read arguments from the test data
        model, starting_media, ingredients_pd, new_round_n, batch_size, \
            sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
            go_beyond_frontier, used_experiments, redo_experiments = test_data
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
            go_beyond_frontier, used_experiments, redo_experiments = test_data
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
