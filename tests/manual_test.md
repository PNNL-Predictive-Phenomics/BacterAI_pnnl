# BacterAI Manual Testing Guide

This document provides a comprehensive list of commands to manually test the BacterAI CLI functionality.

## Prerequisites

1. Activate the bacterai environment:
```bash
conda activate bacterai
# OR if using the local venv:
source env/bin/activate
```

2. Navigate to the project root:
```bash
cd /Users/reyn999/Desktop/Projects/BacterAI/BacterAI_pnnl-1
```

## Test 1: Help Commands

Test that all CLI commands show help information:

```bash
# Main help
bacterai --help

# Subcommand help
bacterai experiment --help
bacterai ingredients --help
bacterai setup --help
bacterai run --help
```

**Expected Result**: Each command should display usage information and available options.

---

## Test 2: Ingredients Command

Test converting ingredient sheets to JSON format:

```bash
# Test with CSV input
bacterai ingredients tests/test_files/inputs/ingredients.csv -o /tmp/test_ingredients_csv.json

# Verify the output
cat /tmp/test_ingredients_csv.json

# Compare with expected output
diff /tmp/test_ingredients_csv.json tests/test_files/outputs/ingredients_csv.json

# Test with Excel input
bacterai ingredients tests/test_files/inputs/ingredients.xlsx -o /tmp/test_ingredients_xlsx.json

# Verify the output
cat /tmp/test_ingredients_xlsx.json

# Compare with expected output
diff /tmp/test_ingredients_xlsx.json tests/test_files/outputs/ingredients_xlsx.json
```

**Expected Result**: Should create JSON files with properly formatted ingredient data matching the expected outputs.

---

## Test 3: Experiment Command

Test converting experiment configuration to JSON:

```bash
# Test Format 1 (single experiment per row)
bacterai experiment tests/test_files/inputs/experiment_config_format_1.xlsx \
  --outfile config.json \
  --verbose

# Test Format 2 (single experiment)
bacterai experiment tests/test_files/inputs/experiment_config_format_2.xlsx \
  --outfile config.json \
  --verbose

# Test Format 2 with multiple experiments
bacterai experiment tests/test_files/inputs/experiment_config_format_2_multiple.xlsx \
  --outfile config.json \
  --verbose

# Compare output with expected
# (Note: paths will differ, but structure should match)
cat /tmp/experiment_*/config.json
diff -u <(jq -S . tests/test_files/outputs/config.json) <(jq -S . /tmp/experiment_1/config.json) || echo "Paths will differ"
```

**Expected Result**: Should create config.json file(s) in experiment directories with proper formatting matching the structure in test_files/outputs/config.json.

---

## Test 4: Setup Command (Interactive)

Test the full interactive setup workflow:

```bash
# This will prompt for user input
# When prompted for ingredients file, use: tests/test_files/inputs/ingredients.csv
bacterai setup tests/test_files/inputs/experiment_config_format_1.xlsx \
  --verbose

# Alternative: Test with multiple experiments
# When prompted:
# - Choose whether to use shared ingredients (Y/N)
# - If shared, provide: tests/test_files/inputs/ingredients.xlsx
bacterai setup tests/test_files/inputs/experiment_config_format_2_multiple.xlsx \
  --verbose
```

**Expected Prompts**:
- Directory existence handling (if directory already exists)
- Use shared ingredients? (Y/N) - for multiple experiments
- Ingredients file path: Enter `tests/test_files/inputs/ingredients.csv` or `.xlsx`
- Sheet name (if Excel): Press Enter to use first sheet

**Expected Result**: Should create both config.json and ingredients.json in experiment directory.

---

## Test 5: Run Command - First Round

Test running the first round of an experiment:

```bash
# Clean up any existing test runs
rm -rf tests/test_experiment/Round*

# Run first round
bacterai run tests/test_experiment --verbose
```

**Expected Result**: 
- Should create `Round1/` directory
- Should contain:
  - `batch_meta_*.csv`
  - `batch_dp_*.csv`
  - `gpr_model/` or `nn_models/` directory
  - `run_metrics.json`
  - `random_train_kickstart*.csv`

**Verification**:
```bash
ls -la tests/test_experiment/Round1/
cat tests/test_experiment/Round1/run_metrics.json
```

---

## Test 6: Feature Extraction (Between Rounds)

Test extracting features from Biotek plate reader data:

```bash
# Extract features from test data
python biotek_feature_extract.py tests/test_experiment \
  -d test_date \
  -r 1 \
  -s 600 \
  -f delta_od
```

**Expected Result**: Should create `mapped_data_test_date_biotek_delta_od_data.csv` in Round1 directory.

**Verification**:
```bash
ls tests/test_experiment/Round1/mapped_data*.csv
head -5 tests/test_experiment/Round1/mapped_data*.csv
```

---

## Test 7: Run Command - Second Round

Test running a subsequent round with existing data:

```bash
# Copy Round_test to Round1 to simulate completed first round
rm -rf tests/test_experiment/Round1
cp -r tests/test_experiment/Round_test tests/test_experiment/Round1

# Run second round
bacterai run tests/test_experiment --verbose
```

**Expected Result**: 
- Should create `Round2/` directory
- Should process Round1 results
- Should generate new batch based on trained model

**Note**: This test is currently known to have issues with flat GPR predictions (see test_second_run skip reason).

---

## Test 8: Plot Only Mode

Test running in plot-only mode (no new experiments):

```bash
# Requires existing Round data
bacterai run tests/test_experiment --plot-only --verbose
```

**Expected Result**: Should regenerate plots without creating new batch files.

---

## Test 9: Automated Tests

Run the full automated test suite:

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/bacterai --cov-report=html

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests only
pytest tests/test_run.py -v              # Run tests
pytest tests/test_feature_extract.py -v  # Feature extraction tests

# View coverage report
open htmlcov/index.html  # macOS
# OR
xdg-open htmlcov/index.html  # Linux
```

**Expected Result**: 
- 44 tests should pass
- 1 test should be skipped (test_second_run)
- Coverage report generated in `htmlcov/`

## Test 10: Configuration Validation

Test the configuration parsing with various input formats:

```bash
# Test different experiment config formats
pytest tests/integration/test_configuration.py -v

# These tests validate:
# - Format 1 vs Format 2 parsing
# - Handling missing paths
# - Handling missing ingredient labels
# - Multiple experiments in one file
# - CSV vs Excel inputs
```

**Expected Result**: All configuration tests should pass, validating proper parsing of different input formats.

---

## Test 11: End-to-End Workflow

Complete workflow from scratch:

```bash
# 1. Clean slate
TEST_EXP=/tmp/bacterai_test_experiment
rm -rf $TEST_EXP
mkdir -p $TEST_EXP

# 2. Copy test configuration
cp tests/test_experiment/config.json $TEST_EXP/
cp tests/test_experiment/ingredients.json $TEST_EXP/

# 3. Update config to point to new location
# (Edit config.json manually or use sed)
sed -i '' "s|tests/test_experiment|$TEST_EXP|g" $TEST_EXP/config.json

# 4. Run first round
bacterai run $TEST_EXP --verbose

# 5. Verify output
ls -la $TEST_EXP/Round1/

# 6. (Optional) Create mock data and run second round
# This would require creating mapped_data file manually

# Cleanup
rm -rf $TEST_EXP
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure bacterai is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **File Not Found**: Check that all paths in config.json are absolute paths

3. **GPR Training Issues**: 
   - Check that fitness values have sufficient variance (std dev > 0.05)
   - Verify training data has at least 10 experiments

4. **Timeout Issues**: 
   - Adjust `timeout_min` in config.json
   - Reduce `batch_size` for faster testing
   - Reduce `n_bags` (for neural nets) or `max_iter` (for GPR)

### Debug Mode

For detailed debugging output:
```bash
# Run with Python directly for full traceback
python -m bacterai.main run tests/test_experiment --verbose

# Or use pdb for interactive debugging
python -m pdb -m bacterai.main run tests/test_experiment
```

---

## Expected Test Coverage

After running all tests, coverage should be approximately:
- Overall: ~20-30%
- `run/` module: ~40-50%
- `ml/` module: ~60-70%
- `utils/` module: ~90-100%

Areas with low coverage (expected):
- `cli.py`: Interactive prompts (difficult to test)
- `configuration.py`: Complex parsing logic (partially tested)
- `sim/core.py`: Simulation engine (complex, needs more tests)
- `main.py`: Entry point (tested via CLI)

---

## Notes

- Some tests may take 5-10 seconds due to model training
- test_second_run is currently skipped due to GPR model prediction issues
- Feature extraction tests require properly formatted Biotek Excel files
- Transfer learning tests are not yet implemented
