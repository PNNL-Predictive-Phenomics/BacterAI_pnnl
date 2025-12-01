# BacterAI Testing Guide

## Test Structure

The BacterAI test suite is organized into multiple levels:

```
tests/
├── unit/                    # Fast, isolated unit tests
│   ├── test_constants.py
│   ├── test_paths.py
│   ├── test_sim_types.py
│   └── test_export.py
├── integration/             # Integration tests for workflows
│   └── test_configuration.py
├── conftest.py             # Shared pytest fixtures
├── test_run.py             # Legacy run tests
├── test_sim.py             # Legacy simulation tests
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
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage report
pytest --cov=src/bacterai --cov-report=html

# Run specific test file
pytest tests/unit/test_constants.py

# Run specific test class or function
pytest tests/unit/test_constants.py::TestConstants::test_colors_defined
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
```

### Test Performance

The BacterAI test suite has been optimized for fast execution:

- **All passing tests** (44 tests, ~10 seconds): Unit, integration, and experiment tests

The test experiment config uses reduced parameters for speed:
- `batch_size`: 10 (vs 100 in production)
- `n_bags`: 3 (vs 25 in production) 
- `n_rollouts`: 1 (vs 2 in production)
- `timeout_min`: 5 (vs 60 in production)

## Test Configuration

The test experiment uses optimized settings for fast execution:
- GPR training: 50 iterations (vs 100 in production)
- Batch generation timeout: 20 minutes (to allow simulations to complete)
- Small batch size (10) and ensemble (3 models)

If tests fail with empty batches, the simulation timeout may need adjustment.

## Writing New Tests

### Unit Tests

Unit tests should:
- Test a single function or method
- Use mocks/stubs for dependencies
- Run quickly (< 1 second each)
- Be independent from other tests

Example:

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
