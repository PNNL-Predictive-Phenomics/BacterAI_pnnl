# BacterAI Testing

Use two testing paths in this repository:

- `pytest` for the automated test suite
- `manual_test.md` for CLI smoke checks and interactive workflows

## Automated Tests

Activate the local environment, then run `pytest` from the repository root.

```bash
source env/bin/activate
pytest
```

Pytest is configured in `pytest.ini`. Running `pytest` already includes:

- verbose output
- short tracebacks
- coverage for `src/bacterai`
- an HTML coverage report in `htmlcov/`

Useful variations:

```bash
# Run one file
pytest tests/cli/test_commands_setup_paths.py

# Run one directory
pytest tests/configuration/

# Run one test
pytest tests/run/test_core.py::TestProcessResults::test_process_results

# Filter by marker
pytest -m "not slow"

# Show print output
pytest -s

# Stop on first failure
pytest -x
```

If you need test dependencies in a fresh environment:

```bash
pip install -e ".[dev]"
```

## Manual Tests

Use [manual_test.md](tests/manual_test.md) when you need to verify behavior that is easier to confirm through the CLI directly, especially:

- help output for CLI commands
- file conversion commands such as `experiment` and `ingredients`
- interactive `setup` flows
- end-to-end `run` behavior, including auto-setup cases
- `process_data` command behavior against sample inputs

Treat the manual guide as a checklist. Run the relevant commands, confirm the expected files are created, and compare outputs against the sample data under `tests/test_files/` and `tests/test_experiment/` where applicable.

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
