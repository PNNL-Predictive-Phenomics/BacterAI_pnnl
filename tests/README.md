# BacterAI Testing Guide

## Test Structure

The BacterAI test suite is organized to mirror the source code structure:

```
tests/
├── configuration/           # Tests for src/bacterai/configuration/
│   ├── test_experiment.py  # Tests for experiment.py (ExperimentConfig)
│   ├── test_plate_readers_biotek.py  # Tests for Biotek reader
│   └── test_plate_readers_tecan.py   # Tests for Tecan reader
├── run/                     # Tests for src/bacterai/run/
│   ├── test_batch.py       # Tests for batch.py
│   ├── test_core.py        # Tests for core.py (process_results, execute_experiment)
│   ├── test_export.py      # Tests for export.py
│   └── test_paths.py       # Tests for paths.py
├── sim/                     # Tests for src/bacterai/sim/
│   └── test_types.py       # Tests for simulation types
├── ml/                      # Tests for src/bacterai/ml/ (placeholder)
├── utils/                   # Tests for src/bacterai/utils/ (placeholder)
├── analysis/                # Tests for src/bacterai/analysis/ (placeholder)
├── test_sim.py             # Legacy simulation tests (kept as reference)
├── test_files/             # Shared test data files
│   ├── tecan_initial.asc   # Sample Tecan data
│   └── tecan_final.asc     # Sample Tecan data
├── test_experiment/        # Sample experiment directory structure
├── conftest.py             # Shared pytest fixtures
└── README.md               # This file
```

## Running Tests

### Install Test Dependencies

```bash
# Activate your virtual environment
source env/bin/activate

# Install dev dependencies (includes pytest)
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run tests for specific modules
pytest tests/configuration/  # All configuration tests
pytest tests/run/            # All run module tests
pytest tests/sim/            # All sim module tests

# Run with coverage report
pytest --cov=src/bacterai --cov-report=html
# Run specific test file
pytest tests/configuration/test_plate_readers_tecan.py
pytest tests/run/test_core.py

# Run specific test class or function
pytest tests/configuration/test_plate_readers_tecan.py::TestTecanReader::test_read_tecan_basic
pytest tests/run/test_core.py::TestProcessResults::test_process_results
pytest tests/run/test_core.py::TestProcessResults::test_process_results
```

### Run Tests with Different Verbosity

```bash
# Quiet mode (only show failures)
pytest -q

# Verbose mode (show all test names)
pytest -v

# Very verbose (show full output)
pytest -vv
```

### Run Tests in Parallel (faster)

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel using multiple CPU cores
pytest -n auto
```

## Test Markers

Tests can be marked with categories:

```python
import pytest

@pytest.mark.slow
def test_large_simulation():
    # This test takes a long time
    pass

@pytest.mark.integration
def test_full_workflow():
    # This is an integration test
    pass
```

Run tests by marker:

```bash
# Skip slow tests (recommended for regular development)
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run only integration tests
pytest -m integration

# Run only unit tests
pytest -m unit
### Test Performance

The BacterAI test suite has been optimized for fast execution:

- **All passing tests** (45 tests, ~10 seconds): Unit tests, integration tests, and experiment tests
  - Configuration tests: 17 (experiment config + plate readers)
  - Run tests: 16 (batch, core, export, paths)
  - Sim tests: 12 (types + legacy)

Test organization mirrors source code:
- `tests/configuration/` → `src/bacterai/configuration/`
- `tests/run/` → `src/bacterai/run/`
- `tests/sim/` → `src/bacterai/sim/`

The test experiment config uses reduced parameters for speed:
- `batch_size`: 10 (vs 100 in production)
- `n_bags`: 3 (vs 25 in production) 
- `n_rollouts`: 1 (vs 2 in production)
- `timeout_min`: 5 (vs 60 in production)
- `n_rollouts`: 1 (vs 2 in production)
- `timeout_min`: 5 (vs 60 in production)

## Test Configuration

### Unit Tests

Unit tests should:
- Be located in the appropriate subdirectory matching the source module
- Test a single function or method
- Use mocks/stubs for dependencies
- Run quickly (< 1 second each)
- Be independent from other tests

Example:

```python
# tests/configuration/test_my_feature.py
import pytest
from bacterai.configuration import my_function

def test_my_function_basic():
    result = my_function(5)
    assert result == 10
```
```python
# tests/unit/test_mymodule.py
import pytest
from src.bacterai.mymodule import my_function

def test_my_function_basic():
    result = my_function(5)
    assert result == 10
```

### Integration Tests

Integration tests should:
- Test multiple components working together
- Use fixtures for setup/teardown
- Test realistic workflows
- May take longer to run

Example:

```python
# tests/integration/test_workflow.py
import pytest

def test_experiment_setup_to_batch(setup_experiment_dir):
    # Test complete workflow from setup to batch creation
    settings, ingredients_pd, ingredients_list = setup_experiment(...)
    batch_df = create_batch(...)
    assert len(batch_df) > 0
```

### Using Fixtures

Fixtures are defined in `conftest.py` and available to all tests:

```python
def test_with_temp_dir(temp_experiment_dir):
    # temp_experiment_dir is automatically created and cleaned up
    config_path = temp_experiment_dir / "config.json"
    assert config_path.parent.exists()
```

## Test Coverage

View coverage report after running tests:

```bash
# Generate HTML coverage report
pytest --cov=src/bacterai --cov-report=html

# Open the report in your browser
open htmlcov/index.html
```

Current coverage goals:
- **Core modules**: > 80% coverage
- **Utilities**: > 90% coverage
- **Integration**: > 70% coverage

## Continuous Integration

Tests should run on:
- Every commit (via pre-commit hook)
- Every pull request (via CI/CD)
- Before releases

## Test Data

Test data is stored in:
- `tests/test_experiment/` - Sample experiment data
- `tests/test_files/` - Sample input files
- `tests/conftest.py` - Generated mock data via fixtures

**Never commit large data files!** Use fixtures to generate mock data instead.

## Debugging Failed Tests

```bash
# Show print statements from tests
pytest -s

# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l
```

## Best Practices

1. **Test names should be descriptive**
   - ✅ `test_export_creates_files_with_correct_names`
   - ❌ `test_export_1`

2. **Use fixtures for setup**
   - Don't repeat setup code in every test
   - Use `conftest.py` for shared fixtures

3. **Keep tests independent**
   - Tests should not depend on execution order
   - Use `temp_experiment_dir` fixture for file operations

4. **Test both success and failure cases**
   - Test normal operation
   - Test error conditions with `pytest.raises()`

5. **Mock external dependencies**
   - Don't call external APIs in tests
   - Use `unittest.mock` or `pytest-mock`

## Common Issues

### Import Errors

If you see import errors, make sure:
1. Virtual environment is activated
2. Package is installed in editable mode: `pip install -e .`
3. Using `src.bacterai.` prefix for imports

### Test Discovery Issues

If pytest doesn't find your tests:
1. Check file names start with `test_`
2. Check class names start with `Test`
3. Check function names start with `test_`
4. Run `pytest --collect-only` to see what pytest finds

### Fixture Issues

If fixtures aren't working:
1. Check `conftest.py` is in the right location
2. Verify fixture function name matches usage
3. Check for typos in fixture names
