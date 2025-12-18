import unittest
import os
import pytest
import pandas as pd

from bacterai.configuration.plate_readers import (
    read_tecan,
    extract_tecan_delta_od,
    process_tecan_data,
    save_mapped_data,
)


class TestTecanReader(unittest.TestCase):
    """Tests for Tecan plate reader functionality."""
    
    def setUp(self):
        """Set up test file paths."""
        # Go up one level from configuration/ to tests/
        tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.test_files_dir = os.path.join(tests_dir, 'test_files')
        self.initial_file = os.path.join(self.test_files_dir, 'tecan_initial.asc')
        self.final_file = os.path.join(self.test_files_dir, 'tecan_final.asc')
    
    def test_read_tecan_basic(self):
        """Test reading a Tecan .asc file."""
        df = read_tecan(self.initial_file)
        
        # Should have 'well' and 'y' columns
        self.assertIn('well', df.columns)
        self.assertIn('y', df.columns)
        
        # Should have 20 wells (A1-D3 based on our test data)
        self.assertEqual(len(df), 20)
        
        # Check some specific values
        a1_row = df[df['well'] == 'A1'].iloc[0]
        self.assertAlmostEqual(a1_row['y'], 0.0446, places=4)
        
        h1_row = df[df['well'] == 'H1'].iloc[0]
        self.assertAlmostEqual(h1_row['y'], 0.0611, places=4)
    
    def test_read_tecan_data_types(self):
        """Test that data types are correct."""
        df = read_tecan(self.initial_file)
        
        # 'well' should be string
        self.assertEqual(df['well'].dtype, object)
        
        # 'y' should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df['y']))
    
    def test_extract_tecan_delta_od(self):
        """Test calculating delta OD from paired Tecan files."""
        delta_df = extract_tecan_delta_od(self.initial_file, self.final_file)
        
        # Should have 'well' and 'feature' columns
        self.assertIn('well', delta_df.columns)
        self.assertIn('feature', delta_df.columns)
        
        # Should have same number of wells
        self.assertEqual(len(delta_df), 20)
        
        # Check delta OD calculation for A1: 0.5446 - 0.0446 = 0.5
        a1_row = delta_df[delta_df['well'] == 'A1'].iloc[0]
        self.assertAlmostEqual(a1_row['feature'], 0.5, places=4)
        
        # Check delta OD calculation for A2: 0.8128 - 0.1128 = 0.7
        a2_row = delta_df[delta_df['well'] == 'A2'].iloc[0]
        self.assertAlmostEqual(a2_row['feature'], 0.7, places=4)
    
    def test_extract_tecan_delta_od_handles_mismatched_wells(self):
        """Test that delta OD only includes wells present in both files."""
        # This tests the merge behavior
        delta_df = extract_tecan_delta_od(self.initial_file, self.final_file)
        
        # All wells should be present in both files in our test data
        self.assertEqual(len(delta_df), 20)
        
        # No NaN values should be present
        self.assertFalse(delta_df['feature'].isna().any())


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_experiment', 'experiment_request', 'test_date')),
    reason="Test experiment directory not found - requires full test setup"
)
class TestTecanProcessing(unittest.TestCase):
    """Integration tests for Tecan data processing pipeline."""
    
    def setUp(self):
        """Set up test experiment paths."""
        # Go up one level from configuration/ to tests/
        tests_dir = os.path.dirname(os.path.dirname(__file__))
        self.path = os.path.join(tests_dir, 'test_experiment')
        self.date = 'test_date'
        self.round_number = '_test'
    
    def test_process_tecan_data_requires_delta_od(self):
        """Test that Tecan processing only supports delta_od feature."""
        with self.assertRaises(NotImplementedError):
            process_tecan_data(self.path, self.date, self.round_number, feature='growth_rate')
        
        with self.assertRaises(NotImplementedError):
            process_tecan_data(self.path, self.date, self.round_number, feature='lag_time')


class TestSaveFunction(unittest.TestCase):
    """Tests for the save_mapped_data function."""
    
    def test_save_mapped_data_filename_format(self):
        """Test that save function creates correct filename."""
        # Create a minimal test DataFrame
        test_df = pd.DataFrame({
            'feature': [0.1, 0.2],
            'bad': [0, 0],
            'plate_control': [False, False],
            'plate_blank': [False, False],
            'parent_plate': ['plate1', 'plate1'],
            'experiment_number': [1, 2],
            'strain': ['test', 'test'],
            'environment': ['aerobic', 'aerobic']
        })
        
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Create Round directory
            round_dir = os.path.join(temp_dir, 'Round1')
            os.makedirs(round_dir)
            
            # Save with tecan reader type
            out_path = save_mapped_data(test_df, temp_dir, 'test_date', 1, 'tecan', 'delta_od')
            
            # Check filename format
            expected_file = 'mapped_data_test_date_tecan_delta_od_data.csv'
            self.assertTrue(out_path.endswith(expected_file))
            
            # Check file exists
            self.assertTrue(os.path.exists(out_path))
            
            # Check file can be read back
            loaded_df = pd.read_csv(out_path)
            self.assertEqual(len(loaded_df), 2)
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
