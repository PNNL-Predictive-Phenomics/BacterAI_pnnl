"""Tests for src/bacterai/run/core.py"""
import unittest
import os
import json
import shutil
import pandas as pd
import pytest
from src.bacterai.run.core import process_results, execute_experiment


exp_path = os.path.join(os.path.dirname(__file__), '..', 'test_experiment')
config_path = os.path.join(exp_path, 'config.json')
round_test_folder = os.path.join(exp_path, "Round_test")


class TestProcessResults(unittest.TestCase):
    """Test the process_results function that processes experimental data."""
    
    @classmethod
    def setUpClass(cls):
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
        """Test that process_results correctly processes experimental data."""
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
        
        # Load and prepare ingredients
        ingredients_full_path = os.path.join(EXPT_FOLDER, INGREDIENTS_FILE)
        with open(ingredients_full_path, "r") as f:
            ingredients_json = json.load(f)
        ingredients_pd = pd.json_normalize(ingredients_json["ingredients"])
        ingredients_pd = ingredients_pd[~(ingredients_pd.INGREDIENT.str.contains("\\:\\:"))]
        ingredients_pd = ingredients_pd[~(ingredients_pd["MIN_VALUE"] == ingredients_pd["MAX_VALUE"])]
        ingredients_pd = ingredients_pd.reset_index(drop=True)
        INGREDIENTS = ingredients_pd["INGREDIENT"].tolist()
        
        # Process results
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
        
        # TODO: Clarify if this is expected output or needs adjustment
        # Assert the output shape (Round_test has 60 experiments)
        self.assertEqual(X_train.shape, (100, 5))
        self.assertEqual(y_train.shape, (100, ))
        self.assertEqual(len(used_experiments), 100)


class TestExecuteExperiment(unittest.TestCase):
    """Test the execute_experiment function that runs complete experiment rounds."""
    
    @classmethod
    def setUpClass(cls):
        # Clean up ALL existing Round folders from previous test runs (except Round_test)
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up ALL Round folders created during tests (except Round_test)
        '''
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        '''
        pass

    def tearDown(self):
        # Clean up Round folders after each individual test (except Round_test)
        '''
        for item in os.listdir(exp_path):
            if item.startswith("Round") and item != "Round_test":
                item_path = os.path.join(exp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        '''
        pass

    def test_first_run(self):
        """Test creating the first round (Round1) from scratch."""
        round_number = 1
        execute_experiment(exp_path, plot_only=False)
        round_folder = os.path.join(exp_path, f"Round{round_number}")
        
        # Check that all expected files were created
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

    #@pytest.mark.skip(reason="Test data issue: GPR model produces flat predictions causing empty batch generation")
    def test_second_run(self):
        """Test creating the second round (Round2) from existing Round1 data."""
        # Copy Round_test to Round1 to simulate completed first round
        round_1_folder = os.path.join(exp_path, "Round1")
        if os.path.exists(round_1_folder):
            shutil.rmtree(round_1_folder)
        shutil.copytree(round_test_folder, round_1_folder)
        
        # Verify Round1 has the mapped_data file
        print(round_1_folder)
        round1_files = os.listdir(round_1_folder)
        print(round1_files)
        mapped_data_exists = any('mapped_data' in f for f in round1_files)
        self.assertTrue(mapped_data_exists, f"Round1 should have mapped_data file. Files: {round1_files}")
        
        # Execute experiment for Round2
        round_number = 2
        execute_experiment(exp_path, plot_only=False)
        round_folder = os.path.join(exp_path, f"Round{round_number}")
        
        # Check that all expected files were created
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
