"""Tests for ExperimentConfig parsing and coercion."""
import pandas as pd

from src.bacterai.configuration.experiment import ExperimentConfig


def test_detect_format_vertical():
    df = pd.DataFrame({"key": ["experiment_path", "batch_size"], "value": ["/tmp/exp", 96]})
    assert ExperimentConfig.detect_format(df) == "vertical"


def test_detect_format_horizontal():
    df = pd.DataFrame({"experiment_path": ["/tmp/exp"], "batch_size": [96]})
    assert ExperimentConfig.detect_format(df) == "horizontal"


def test_parse_configs_vertical_basic(temp_experiment_dir):
    df = pd.DataFrame({"key": ["experiment_path", "batch_size", "use_unique"], "value": [str(temp_experiment_dir), 96, "true"]})
    path = temp_experiment_dir / "exp_vertical.csv"
    df.to_csv(path, index=False)

    cfgs = ExperimentConfig.parse_configs(str(path), sheet=None, force_format="vertical")
    assert cfgs[0]["batch_size"] == 96
    assert cfgs[0]["use_unique"] is True


def test_parse_configs_horizontal_basic(temp_experiment_dir):
    df = pd.DataFrame({
        "experiment_path": [str(temp_experiment_dir)],
        "batch_size": [96],
        "redo_threshold": ["0,1"],
    })
    path = temp_experiment_dir / "exp_horizontal.csv"
    df.to_csv(path, index=False)

    cfgs = ExperimentConfig.parse_configs(str(path), sheet=None, force_format="horizontal")
    assert cfgs[0]["batch_size"] == 96
    assert cfgs[0]["redo_threshold"] == [0.0, 1.0]


def test_parse_configs_vertical_ignores_placeholder_header_for_experiment_path(temp_experiment_dir):
    path = temp_experiment_dir / "exp_vertical_placeholder.csv"
    path.write_text("experiment_path,\ningredients_file,\nbatch_size,96\n")

    cfgs = ExperimentConfig.parse_configs(str(path), sheet=None, force_format="vertical")

    assert len(cfgs) == 1
    assert cfgs[0]["experiment_path"] is None
    assert cfgs[0]["batch_size"] == 96
