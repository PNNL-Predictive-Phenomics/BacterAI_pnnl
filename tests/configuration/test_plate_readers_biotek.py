import unittest
import os
import pytest
from unittest.mock import patch
from io import StringIO
from bacterai.configuration.plate_readers import read_biotek, process_biotek_data, save_mapped_data

# Set up input arguments - go up one level from configuration/ to tests/
tests_dir = os.path.dirname(os.path.dirname(__file__))
path = os.path.join(tests_dir, 'test_experiment')
date = 'test_date'
round_number = '_test'
signal = 600
feature = 'delta_od'
data_file_path = os.path.join(path, "experiment_request", date, 'data', 'testfid.xlsx')

@pytest.mark.skipif(not os.path.exists(data_file_path), 
                    reason="Test Excel file not found - biotek feature tests require sample data")
class TestBiotekFeatureExtract(unittest.TestCase):
    def test_read_biotek(self):
        # Construct path to test data file
        data_file_path = os.path.join(path, "experiment_request", date, 'data', 'testfid.xlsx')
        biotek_df = read_biotek(data_file_path)
        # Asset the output shape
        self.assertEqual(biotek_df.shape, (1152, 4))

    def test_biotek_feature_extract(self):
        # Process the data and save
        result_df = process_biotek_data(path, date, round_number, signal, feature)
        out_path = save_mapped_data(result_df, path, date, round_number, 'biotek', feature)
        
        # Verify the file was created
        self.assertTrue(os.path.exists(out_path), f"Output file should exist: {out_path}")
        
        # Verify the output filename
        round_folder = os.path.join(path, f"Round{round_number}")
        out_file = 'mapped_data_' + date + '_biotek_' + feature + '_data.csv'
        expected_path = os.path.join(round_folder, out_file)
        self.assertEqual(out_path, expected_path)