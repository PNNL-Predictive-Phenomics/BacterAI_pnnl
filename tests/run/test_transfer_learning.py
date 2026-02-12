"""Tests for transfer learning utilities."""
import pandas as pd
from types import SimpleNamespace

from src.bacterai.run.transfer_learning import handle_data_dir_transition, load_pretrained_model
from src.bacterai.utils.constants import AA_SHORT, BASE_NAMES
from src.bacterai.run.models import ModelType


def _write_df(path, df):
    df.to_csv(path, index=False)


def test_handle_data_dir_transition_combines_data(temp_experiment_dir):
    n_aa = len(AA_SHORT)
    n_base = len(BASE_NAMES)
    n_ingredients = n_aa + n_base

    transfer_path = temp_experiment_dir / "train_pred_transfer.csv"
    new_round_folder = temp_experiment_dir / "Round2"
    new_round_folder.mkdir()

    transfer_df = pd.DataFrame(
        [[1] * n_aa + [0.1, 0.2, 0.3]],
        columns=AA_SHORT + ["y_true", "y_pred", "y_pred_var"],
    )
    _write_df(transfer_path, transfer_df)

    round1_df = pd.DataFrame(
        [[1] * n_ingredients + [0.4, 0.5, 0.6]],
        columns=AA_SHORT + BASE_NAMES + ["y_true", "y_pred", "y_pred_var"],
    )
    _write_df(new_round_folder / "train_pred.csv", round1_df)

    settings = SimpleNamespace(
        transfer_data_dir=str(transfer_path),
        aas_only=False,
        round_number=2,
    )

    X_train, y_train = handle_data_dir_transition(settings, str(new_round_folder), n_ingredients)

    assert X_train.shape[1] == n_ingredients
    assert len(y_train) == 2
    assert (new_round_folder / "train_pred_orig.csv").exists()


def test_load_pretrained_model_none():
    settings = SimpleNamespace(transfer_model_folder=None, model_type=ModelType.GPR)
    assert load_pretrained_model(settings) is None


def test_load_pretrained_model_dispatch(monkeypatch, temp_experiment_dir):
    settings = SimpleNamespace(transfer_model_folder=str(temp_experiment_dir), model_type=ModelType.GPR)

    class DummyModel:
        pass

    def _fake_load(path):
        assert path == str(temp_experiment_dir)
        return DummyModel()

    from src.bacterai.run import models as model_module
    monkeypatch.setattr(model_module.GPRModel, "load_trained_models", staticmethod(_fake_load))

    loaded = load_pretrained_model(settings)
    assert isinstance(loaded, DummyModel)
