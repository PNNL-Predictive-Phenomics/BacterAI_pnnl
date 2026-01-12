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

## Overview of CLI Commands

BacterAI provides five main commands:

1. **`bacterai experiment`** - Convert experiment config files to JSON (creates config.json only)
2. **`bacterai ingredients`** - Convert ingredient files to JSON (creates ingredients.json only)
3. **`bacterai setup`** - Interactive setup (creates both config.json AND ingredients.json)
4. **`bacterai run`** - Execute experiment rounds (auto-triggers setup if files missing)
5. **`bacterai process_data`** - Process plate reader data (creates mapped_data CSV)

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
bacterai process_data --help
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

## Test 5a: Run Command - Auto-Setup (NEW)

**Test the new auto-setup feature when config.json or ingredients.json are missing:**

### Scenario 1: Both files missing

```bash
# Create empty experiment directory
mkdir -p /tmp/auto_setup_test
cd /tmp/auto_setup_test

# Run bacterai run (should trigger interactive setup)
bacterai run .
```

**When prompted:**
1. "Provide experiment config filepath (CSV/XLSX):" 
   - Enter: `tests/test_files/inputs/experiment_config_format_1.xlsx`
2. "Excel sheet name or index (press Enter to use first sheet):" 
   - Press Enter
3. If path mismatch detected, choose option 2 (use run path)
4. "Enter ingredients file path (CSV/XLSX):" 
   - Enter: `tests/test_files/inputs/ingredients.csv`
5. Press Enter for sheet

**Expected Result**:
- Creates `config.json` and `ingredients.json` automatically
- Continues with Round 1 execution
- Output: "Setup complete! Continuing with experiment run..."

**Verification**:
```bash
ls -la /tmp/auto_setup_test/
cat /tmp/auto_setup_test/config.json
cat /tmp/auto_setup_test/ingredients.json
ls -la /tmp/auto_setup_test/Round1/
```

### Scenario 2: Only ingredients.json missing

```bash
# Setup: Create directory with only config.json
mkdir -p /tmp/partial_setup_test
cp tests/test_experiment/config.json /tmp/partial_setup_test/

# Run (should only prompt for ingredients)
bacterai run /tmp/partial_setup_test
```

**When prompted:**
- Only asked for ingredients file (not config)
- If config path differs and has ingredients, offered to copy it

**Expected Result**: Creates missing ingredients.json and proceeds

### Scenario 3: Path conflict resolution

```bash
# This tests when config specifies one path but you run in another
mkdir -p /tmp/run_path_test

# Run with config that specifies different path
bacterai run /tmp/run_path_test
# Provide: tests/test_files/inputs/experiment_config_format_1.xlsx
```

**Expected Prompts**:
```
WARNING: Path mismatch detected
Config file specifies: [path from config]
You are running in:    /tmp/run_path_test

Which path should be used?
  1. Use config path (create/update files there)
  2. Use run path (override config, use current directory)
```

**Test both options**:
- Option 1: Creates files at config path
- Option 2: Creates files at /tmp/run_path_test

### Scenario 4: Multiple experiments in config

```bash
mkdir -p /tmp/multi_exp_test
bacterai run /tmp/multi_exp_test
# Provide: tests/test_files/inputs/experiment_config_format_2_multiple.xlsx
```

**Expected Prompt**:
```
Found 3 experiments in config file:
  1. Experiment_A (path: ...)
  2. Experiment_B (path: ...)
  3. Experiment_C (path: ...)
Enter number:
```

**Expected Result**: User selects which experiment to set up

### Scenario 5: Smart ingredients reuse (NEW)

```bash
# Setup: Create original experiment with ingredients
mkdir -p /tmp/original_exp
cp tests/test_experiment/config.json /tmp/original_exp/
cp tests/test_experiment/ingredients.json /tmp/original_exp/

# Create new empty experiment directory
mkdir -p /tmp/new_exp

# Run with config pointing to original (requires manual config edit or special test file)
# When path mismatch occurs and you choose config path, if ingredients exists there:
```

**Expected Prompt**:
```
✓ Found existing ingredients.json at:
  /tmp/original_exp/ingredients.json
Copy and use this ingredients file? (Y/N):
```

**Expected Result**: If Y, copies ingredients without prompting for file

---

## Test 6: Process Data Command (Plate Reader Feature Extraction)

Test extracting features from plate reader data using the new `process_data` command:

### Biotek Data Processing

```bash
# Extract features from Biotek plate reader data
bacterai process_data biotek tests/test_experiment \
  -d test_date \
  -r 1 \
  -s 600 \
  -f delta_od

# With verbose output
bacterai process_data biotek tests/test_experiment \
  -d test_date \
  -r 1 \
  -s 600 \
  -f delta_od \
  --verbose

# Using current directory as path (when inside experiment directory)
cd tests/test_experiment
bacterai process_data biotek . -d test_date -r 1 -s 600 -f delta_od
cd ../..

# Custom output location
bacterai process_data biotek tests/test_experiment \
  -d test_date \
  -r 1 \
  -s 600 \
  -f delta_od \
  -o /tmp/custom_mapped_data.csv
```

**Expected Result**: Should create `mapped_data_test_date_biotek_delta_od_data.csv` in Round1 directory.

### Tecan Data Processing

```bash
# Extract features from Tecan plate reader data
# Note: --signal is not used for Tecan (will be ignored if provided)
bacterai process_data tecan tests/test_experiment \
  -d test_date \
  -r 1 \
  -f delta_od

# With verbose output
bacterai process_data tecan tests/test_experiment \
  -d test_date \
  -r 1 \
  -f delta_od \
  --verbose
```

**Expected Result**: Should create `mapped_data_test_date_tecan_delta_od_data.csv` in Round1 directory.

### Error Validation Tests

```bash
# Test missing --signal for Biotek (should fail with helpful error)
bacterai process_data biotek tests/test_experiment -d test_date -r 1 -f delta_od

# Test invalid reader type (should fail)
bacterai process_data invalid_reader tests/test_experiment -d test_date -r 1 -f delta_od

# Test non-existent directory (should fail)
bacterai process_data biotek /nonexistent/path -d test_date -r 1 -s 600 -f delta_od

# Test missing experiment_request directory (should fail with helpful error)
mkdir /tmp/empty_exp
bacterai process_data biotek /tmp/empty_exp -d test_date -r 1 -s 600 -f delta_od
rm -rf /tmp/empty_exp
```

### Verification

```bash
# Check output file was created
ls tests/test_experiment/Round1/mapped_data*.csv

# View first few rows
head -5 tests/test_experiment/Round1/mapped_data*.csv

# Check column structure
head -1 tests/test_experiment/Round1/mapped_data*.csv

# Count experiments processed
wc -l tests/test_experiment/Round1/mapped_data*.csv
```

**Expected Output Structure**:
- Columns: `feature, bad, plate_control, plate_blank, parent_plate, experiment_number, strain, environment`
- ~84 experiments for test data (including controls and blanks)

### Alternative: Legacy Feature Extraction Script

The old method using the standalone script still works but is deprecated:

```bash
# Old method (still functional but prefer using 'bacterai process_data')
python biotek_feature_extract.py tests/test_experiment \
  -d test_date \
  -r 1 \
  -s 600 \
  -f delta_od
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

## Test 7a: Run Command - Auto-Processing (NEW)

**Test the new auto-processing feature that automatically processes plate reader data when mapped_data is missing:**

### Scenario 1: Missing mapped_data after Round 1

```bash
# Setup: Complete Round 1 with plate reader data
cd tests/test_experiment
bacterai run . --verbose  # Complete Round 1

# Add plate reader files (simulating data collection)
# Copy test plate reader files to Round1/data/
mkdir -p Round1/data
cp ../test_files/inputs/plate_reader_*.xlsx Round1/data/

# Remove mapped_data to trigger auto-processing
rm -rf Round1/mapped_data

# Run Round 2 (should auto-process plate reader data)
bacterai run . --verbose
```

**Expected Output**:
```
INFO: mapped_data not found, attempting to auto-process plate reader data...
INFO: Found plate reader data in Round1/data
INFO: Trying signal 600...
INFO: Successfully processed with signal 600
INFO: Created Round1/mapped_data/
```

**Expected Result**:
- Automatically detects missing mapped_data
- Tries processing with common signals (600, 650, 700, 595, 562)
- Creates Round1/mapped_data/ directory
- Continues with Round 2 execution

**Verification**:
```bash
# Check that mapped_data was created
ls -la tests/test_experiment/Round1/mapped_data/

# Verify processing detected correct signal
cat tests/test_experiment/Round1/mapped_data/*.csv | head

# Check Round 2 was created successfully
ls -la tests/test_experiment/Round2/
```

### Scenario 2: Auto-processing with different plate reader formats

```bash
# Test with Biotek format
mkdir -p /tmp/biotek_test/Round1/data
cp tests/test_files/inputs/biotek_*.xlsx /tmp/biotek_test/Round1/data/

# Setup experiment files
cp tests/test_experiment/config.json /tmp/biotek_test/
cp tests/test_experiment/ingredients.json /tmp/biotek_test/

# Run (should auto-process Biotek data)
bacterai run /tmp/biotek_test --verbose
```

**Expected Result**: Processes Biotek Excel time-series format

```bash
# Test with Tecan format
mkdir -p /tmp/tecan_test/Round1/data
cp tests/test_files/inputs/tecan_*.txt /tmp/tecan_test/Round1/data/

cp tests/test_experiment/config.json /tmp/tecan_test/
cp tests/test_experiment/ingredients.json /tmp/tecan_test/

bacterai run /tmp/tecan_test --verbose
```

**Expected Result**: Processes Tecan ASCII endpoint format

### Scenario 3: Auto-processing with signal detection

```bash
# Setup Round 1 with plate reader data at OD650
mkdir -p /tmp/signal_test/Round1/data
# Use test file with OD650 data
cp tests/test_files/inputs/plate_od650_*.xlsx /tmp/signal_test/Round1/data/

cp tests/test_experiment/config.json /tmp/signal_test/
cp tests/test_experiment/ingredients.json /tmp/signal_test/

# Run (should try 600, then 650, then succeed)
bacterai run /tmp/signal_test --verbose
```

**Expected Output**:
```
INFO: Trying signal 600...
WARNING: No data found for signal 600
INFO: Trying signal 650...
INFO: Successfully processed with signal 650
```

### Scenario 4: Auto-processing failure handling

```bash
# Setup Round 1 with no plate reader data
mkdir -p /tmp/no_data_test/Round1/data
# Empty data directory

cp tests/test_experiment/config.json /tmp/no_data_test/
cp tests/test_experiment/ingredients.json /tmp/no_data_test/

# Run (should fail gracefully)
bacterai run /tmp/no_data_test --verbose
```

**Expected Output**:
```
ERROR: Could not process plate reader data automatically
ERROR: No plate reader files found in Round1/data/
Please run 'bacterai process_data' manually or check data files
```

### Scenario 5: Combined auto-setup + auto-processing

```bash
# Test both features together: missing config AND missing mapped_data
mkdir -p /tmp/full_auto_test/Round1/data
cp tests/test_files/inputs/plate_reader_*.xlsx /tmp/full_auto_test/Round1/data/

# Run from scratch (triggers auto-setup first)
cd /tmp/full_auto_test
bacterai run .
```

**When prompted:**
1. Provide config file
2. Provide ingredients file

**Expected Result**:
1. First: Auto-setup creates config.json and ingredients.json
2. Then: Auto-processing detects missing mapped_data and processes it
3. Finally: Continues with Round 2 execution

**Verification**:
```bash
# All files should exist
ls -la /tmp/full_auto_test/
ls -la /tmp/full_auto_test/Round1/mapped_data/
ls -la /tmp/full_auto_test/Round2/
```

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

5. **Auto-Processing Issues (NEW)**:
   - Verify plate reader files exist in `RoundN/data/` directory
   - Check file format matches reader type (Biotek: Excel, Tecan: ASCII/Excel)
   - If auto-processing fails, manually run: `bacterai process_data <reader_type> <path> -d <date> -r <round> -s <signal> -f <feature>`
   - Common signals: 600, 650, 700, 595, 562
   - Check that plate_maps and worklists exist in experiment_request directory

6. **Auto-Setup Issues (NEW)**:
   - Ensure config file has valid experiment_request structure with data/, plate_maps/, and worklists/ directories
   - If path mismatch occurs, choose appropriate option based on your preference
   - For multiple experiments in config, select the correct experiment number when prompted
   - If ingredients reuse fails, verify ingredients.json exists in the config path

### Debug Mode

For detailed debugging output:
```bash
# Run with Python directly for full traceback
python -m bacterai.main run tests/test_experiment --verbose

# Or use pdb for interactive debugging
python -m pdb -m bacterai.main run tests/test_experiment
```

### New CLI Structure (v2.0+)

The CLI has been refactored into a modular structure:
- `src/bacterai/cli/commands.py`: All command handlers
- `src/bacterai/cli/prompts.py`: User interaction functions
- This prevents circular imports and improves maintainability

If encountering import errors, ensure you're using the latest version:
```bash
pip install -e . --force-reinstall --no-cache-dir
```

---

## Expected Test Coverage

After running all tests, coverage should be approximately:
- Overall: ~44% (42 passed, 3 skipped)
- `run/` module: ~40-50%
- `ml/` module: ~60-70%
- `utils/` module: ~90-100%
- `cli/` module: ~30-40% (interactive prompts difficult to test)

Areas with low coverage (expected):
- `cli/prompts.py`: Interactive prompts (difficult to automate)
- `configuration.py`: Complex parsing logic (partially tested)
- `sim/core.py`: Simulation engine (complex, needs more tests)
- `main.py`: Entry point (tested via CLI)

---

## New Features Summary (v2.0)

### Auto-Setup Feature
- **Trigger**: Running `bacterai run` when config.json or ingredients.json are missing
- **Behavior**: 
  - Interactively prompts for config file path
  - Handles multiple experiments in config (user selects which one)
  - Detects path mismatches and prompts for resolution
  - Checks for existing ingredients.json in config path and offers to reuse
  - Creates both config.json and ingredients.json before continuing
- **Test**: See Test 5a scenarios

### Auto-Processing Feature
- **Trigger**: Running `bacterai run` for Round 2+ when Round N-1 has no mapped_data
- **Behavior**:
  - Automatically detects missing mapped_data directory
  - Searches for plate reader files in RoundN-1/data/
  - Tries common signals (600, 650, 700, 595, 562) until success
  - Creates mapped_data directory and continues execution
  - Fails gracefully with helpful message if no data found
- **Test**: See Test 7a scenarios

### CLI Refactoring
- **Change**: Split monolithic cli.py into modular cli/ directory
- **Structure**:
  - `cli/commands.py`: Command handlers (experiment, ingredients, setup, run, process_data)
  - `cli/prompts.py`: User interaction functions
  - `cli/__init__.py`: Exports for external use
- **Benefit**: Eliminates circular imports, improves maintainability

### Smart Ingredients Reuse
- **Feature**: When setting up new experiment, checks if ingredients.json exists in config path
- **Behavior**: Prompts user to copy existing ingredients instead of re-entering
- **Test**: See Test 5a, Scenario 5

---

## Notes

- Some tests may take 5-10 seconds due to model training
- test_second_run is currently skipped due to GPR model prediction issues
- Feature extraction tests require properly formatted Biotek Excel files
- Transfer learning tests are not yet implemented
- Auto-processing tries signals in order: 600, 650, 700, 595, 562
- Auto-setup handles both missing files and path conflicts gracefully
