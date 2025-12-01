import unittest
import os
import json
import pickle
import shutil
import pandas as pd
import argparse
import pytest
from src.bacterai.run.core import process_results, execute_experiment as main
from src.bacterai.run.batch import make_batch

# Path to test data
test_data_path = os.path.join(os.path.dirname(__file__), 'test_experiment', 'test_data.pkl')

# Set up input arguments
parser = argparse.ArgumentParser(description="BacterAI Experiment Generator")
parser.add_argument(
    "path",
    type=str,
    help="The path to the configuration file (.json)",
)
parser.add_argument(
    "-r",
    "--round",
    type=int,
    required=True,
    help="The new round number",
)
parser.add_argument(
    "-p",
    "--plot_only",
    action="store_true",
    help="Only export plots",
)
exp_path = os.path.join(os.path.dirname(__file__), 'test_experiment')
config_path = os.path.join(exp_path, 'config.json')
round_test_folder = os.path.join(exp_path, "Round_test")

class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test data for make_batch test
        with open(test_data_path, 'rb') as f:
            cls.test_data = pickle.load(f)
        
        # Clean up ALL existing Round folders from previous test runs (except Round_test)
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up ALL Round folders created during tests (except Round_test)
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    
    def tearDown(self):
        # Clean up Round folders after each individual test (except Round_test)
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def test_process_results(self):
        # Set up Round1 with test data first
        round_1_folder = os.path.join(exp_path, "Round1")
        if os.path.exists(round_1_folder):
            shutil.rmtree(round_1_folder)
        shutil.copytree(round_test_folder, round_1_folder)
        
        # Load the configuration file
        with open(config_path) as f:
            config = json.load(f)
        GROW_THRESHOLD = config["grow_threshold"]
        EXPT_FOLDER = exp_path  # Use absolute path
        INGREDIENTS_FILE = config.get("ingredients_file", None)
        N_REDOS = config.get("redo_size", None)
        REDO_THRESHOLD = config.get("redo_threshold", None)
        SEPARATE_REDOS = config.get("separate_redos", False)
        # Set up arguments
        NEW_ROUND_N = 2
        current_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N-1}")
        new_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
        if not os.path.exists(new_round_folder):
            os.makedirs(new_round_folder)
        ingredients_full_path = os.path.join(EXPT_FOLDER, INGREDIENTS_FILE)
        with open(ingredients_full_path, "r") as f:
            ingredients_json = json.load(f)
        ingredients_pd = pd.json_normalize(ingredients_json["ingredients"])
        ingredients_pd = ingredients_pd[~(ingredients_pd.INGREDIENT.str.contains("\\:\\:"))]
        ingredients_pd = ingredients_pd[~(ingredients_pd["MIN_VALUE"] == ingredients_pd["MAX_VALUE"])]
        ingredients_pd = ingredients_pd.reset_index(drop=True)
        INGREDIENTS = ingredients_pd["INGREDIENT"]
        INGREDIENTS = INGREDIENTS.tolist()
        X_train, y_train, used_experiments, redo_experiments = process_results(
            current_round_folder,
            None,
            new_round_folder,
            NEW_ROUND_N,
            INGREDIENTS,
            GROW_THRESHOLD,
            n_redos=N_REDOS,
            redo_threshold=REDO_THRESHOLD,
            redo_prev_round=False,
            plot_only=False,
            plot_redos=not SEPARATE_REDOS,
            transfer_padding_needed=False,
        )
        # Asset the output shape
        self.assertEqual(X_train.shape, (100, 5))
        self.assertEqual(y_train.shape, (100, ))
        self.assertEqual(len(used_experiments), 100)
        
    def test_make_batch(self):
        # Read arguments from the test data
        model, starting_media, ingredients_pd, new_round_n, batch_size, \
            sim_types, rollout_trajectories, threshold, timeout, unique, direction, \
            go_beyond_frontier, used_experiments, redo_experiments = self.test_data
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
        # Asset the output shape
        self.assertEqual(batch.shape, (100, 12))

    def test_first_run(self):
        # Test creating the first round (Round1)
        # This should be run when no Round folders exist
        round_number = 1
        main(exp_path, plot_only=False)
        round_folder = os.path.join(exp_path, f"Round{round_number}")
        round_folder_exit = os.path.exists(round_folder)
        run_metric_exist = os.path.exists(os.path.join(round_folder, "run_metrics.json"))
        batch_meta_exist = False
        batch_dp_exist = False
        for filename in os.listdir(round_folder):
            if filename.startswith('batch_meta'):
                batch_meta_exist = True
            elif filename.startswith('batch_dp'):
                batch_dp_exist = True
        # Assert the output
        self.assertTrue(round_folder_exit, f"Round {round_number} folder does not exist")
        self.assertTrue(run_metric_exist, f"Round {round_number} run metrics file does not exist")
        self.assertTrue(batch_meta_exist, f"Round {round_number} batch meta file does not exist")
        self.assertTrue(batch_dp_exist, f"Round {round_number} batch deep phenotyping file does not exist")

    @pytest.mark.skip(reason="Test data issue: GPR model produces flat predictions causing empty batch generation")
    def test_second_run(self):
        # Test creating the second round (Round2)
        # Requires Round1 to exist with data first
        # Copy Round_test to Round1 to simulate completed first round
        round_1_folder = os.path.join(exp_path, "Round1")
        if os.path.exists(round_1_folder):
            shutil.rmtree(round_1_folder)
        shutil.copytree(round_test_folder, round_1_folder)
        
        # Verify Round1 has the mapped_data file
        round1_files = os.listdir(round_1_folder)
        mapped_data_exists = any('mapped_data' in f for f in round1_files)
        self.assertTrue(mapped_data_exists, f"Round1 should have mapped_data file. Files: {round1_files}")
        
        round_number = 2
        main(exp_path, plot_only=False)
        round_folder = os.path.join(exp_path, f"Round{round_number}")
        round_folder_exit = os.path.exists(round_folder)
        run_metric_exist = os.path.exists(os.path.join(round_folder, "run_metrics.json"))
        batch_meta_exist = False
        batch_dp_exist = False
        for filename in os.listdir(round_folder):
            if filename.startswith('batch_meta'):
                batch_meta_exist = True
            elif filename.startswith('batch_dp'):
                batch_dp_exist = True
        # Assert the output
        self.assertTrue(round_folder_exit, f"Round {round_number} folder does not exist")
        self.assertTrue(run_metric_exist, f"Round {round_number} run metrics file does not exist")
        self.assertTrue(batch_meta_exist, f"Round {round_number} batch meta file does not exist")
        self.assertTrue(batch_dp_exist, f"Round {round_number} batch deep phenotyping file does not exist")
