import unittest
import os
import pytest
from unittest.mock import patch
from io import StringIO
from biotek_feature_extract import read_biotek, main

# Set up input arguments
path = os.path.join(os.path.dirname(__file__), 'test_experiment')
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

    @patch('sys.stdout', new_callable=StringIO)
    def test_biotek_feature_extract(self, mock_stdout):
        # Call the main function with the simulated arguments
        main(path, date, round_number, signal, feature)
        round_folder = os.path.join(path, f"Round{round_number}")
        out_file = 'mapped_data_' + date + '_biotek_' + feature + '_data.csv'
        # Assert the output
        self.assertEqual(mock_stdout.getvalue().strip(), f"Successfully wrote Biotek output:\r\n  Location: {round_folder}\r\n  File: {out_file}")