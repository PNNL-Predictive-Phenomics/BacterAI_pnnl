"""Transfer learning utilities for BacterAI experiments."""

import os
import re
import shutil
import pickle
import sys
import types
import pandas as pd
from ..analysis import processing as utils
from ..scripts.size_n_to_m_conversion import fill_new_ingredients
from ..utils.constants import AA_SHORT, BASE_NAMES


class _TransferForestClassifierAdapter:
    """Provide a TIMEDClassifierRF-like predict API for raw TransferForest pickles."""

    def __init__(self, model):
        self._model = model
        prediction_mode = getattr(model, "_prediction_mode", None)
        mode_name = getattr(prediction_mode, "name", str(prediction_mode))
        self._is_classification = mode_name == "CLASSIFICATION"

        # R-side ensemble weights often do not survive round-trip serialization.
        ensemble_weights = getattr(model, "ensemble_weights", None)
        if isinstance(ensemble_weights, list):
            model.ensemble_weights = [None] * len(ensemble_weights)

    def predict(self, X):
        preds_dict = self._model.generate_predictions([X])[0]

        key = next(
            (k for k in ("pred_ensemble_full", "pred_ensemble") if k in preds_dict),
            None,
        )
        if key is None:
            transfer_keys = sorted(k for k in preds_dict if re.match(r"^pred_\d+$", k))
            key = transfer_keys[-1] if transfer_keys else next(iter(preds_dict))

        values = preds_dict[key]
        if self._is_classification:
            return [int(v) for v in values]
        return [float(v) for v in values]


def _ensure_predictable_transfer_rf(loaded_obj):
    """Return an object exposing predict(X) for timed-hpc transfer RF bridge."""
    if hasattr(loaded_obj, "predict"):
        return loaded_obj
    if hasattr(loaded_obj, "generate_predictions"):
        return _TransferForestClassifierAdapter(loaded_obj)
    raise TypeError(
        "Unsupported transfer_rf_pkl object: expected a classifier with predict(X) "
        "or a TransferForest-like object with generate_predictions(...)."
    )


def _extract_transfer_rf_feature_names(classifier):
    """Best-effort extraction of feature names from transfer RF internals."""
    model_obj = getattr(classifier, "_model", classifier)
    source_models = getattr(model_obj, "source_models", None)
    if not source_models:
        return []

    try:
        importance = source_models[0].rx2("importance")
        return list(importance.rownames)
    except Exception:
        return []


def handle_data_dir_transition(settings, new_round_folder, n_ingredients):
    """
    Handle the special transfer learning case for round 2 when combining
    AA-only transfer data with base ingredients from round 1.
    
    Parameters
    ----------
    settings : Settings
        Experiment settings object
    new_round_folder : Path
        Path to the new round folder
    n_ingredients : int
        Total number of ingredients
        
    Returns
    -------
    tuple
        (X_train, y_train) arrays for training, or (None, None) if no transfer learning
    """
    # Check if this is the transfer learning transition case
    if not (settings.transfer_data_dir is not None and 
            not settings.aas_only and 
            settings.round_number == 2):
        return None, None
    
    print("Handling transfer learning data directory transition for round 2...")
    
    # Validate transfer data path
    if "train_pred" not in settings.transfer_data_dir:
        raise Exception("transfer_data_dir must point to a 'train_pred' CSV.")
    
    # Load transfer data (AA-only)
    transfer_data = utils.normalize_ingredient_names(
        pd.read_csv(settings.transfer_data_dir, index_col=None)
    )
    
    # Expand transfer data dimensions by adding base ingredient columns
    transfer_data = fill_new_ingredients(
        transfer_data,
        original_size=len(AA_SHORT),
        fill_column_names=list(
            range(len(AA_SHORT), len(AA_SHORT) + len(BASE_NAMES))
        ),
        fill_on_right=True,
    )
    
    # Load Round 1 data (base ingredients only, already padded with AA columns)
    round_one_data_path = os.path.join(new_round_folder, "train_pred.csv")
    round_one_data = utils.normalize_ingredient_names(
        pd.read_csv(round_one_data_path, index_col=None)
    )
    
    # Standardize column names for merging
    cols_new = list(range(n_ingredients)) + ["y_true", "y_pred", "y_pred_var"]
    transfer_data.columns = round_one_data.columns = cols_new
    
    # Combine the datasets
    combined_data = pd.concat(
        [transfer_data, round_one_data], axis=0, ignore_index=True
    )
    
    # Backup original Round 1 data
    backup_path = os.path.join(new_round_folder, "train_pred_orig.csv")
    shutil.copyfile(round_one_data_path, backup_path)
    
    # Save combined data
    combined_data.to_csv(round_one_data_path, index=None)
    
    # Extract training arrays
    X_train = combined_data.iloc[:, :n_ingredients].to_numpy()
    y_train = combined_data.loc[:, "y_true"].to_numpy()
    
    print(f"Combined {len(transfer_data)} transfer experiments with {len(round_one_data)} round 1 experiments")
    
    return X_train, y_train


def load_pretrained_model(settings):
    """
    Load a pre-trained model for transfer learning.
    
    Parameters
    ----------
    settings : Settings
        Experiment settings object
        
    Returns
    -------
    Model or None
        Loaded model if transfer_model_folder is specified, None otherwise
    """
    if settings.transfer_model_folder is None and settings.transfer_rf_pkl is None:
        return None

    def _bootstrap_omicstl_namespace():
        """Make omicstl importable from local timed-hpc source without running package __init__."""
        if "omicstl" in sys.modules:
            return

        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(this_dir, "..", "..", "..", "timed-hpc", "src", "omicstl"),
            os.path.join(os.getcwd(), "timed-hpc", "src", "omicstl"),
        ]
        for candidate in candidates:
            pkg_dir = os.path.abspath(candidate)
            if os.path.isdir(pkg_dir):
                pkg = types.ModuleType("omicstl")
                pkg.__path__ = [pkg_dir]
                sys.modules["omicstl"] = pkg
                return

    if settings.transfer_rf_pkl is not None:
        print(f"Loading timed-hpc transfer RF model from '{settings.transfer_rf_pkl}'")
        from .models import TimedTransferRFModel

        if not os.path.isfile(settings.transfer_rf_pkl):
            raise FileNotFoundError(
                f"timed-hpc RF pickle not found: {settings.transfer_rf_pkl}"
            )

        try:
            import importlib

            _bootstrap_omicstl_namespace()
            timed_classifiers = importlib.import_module("omicstl.classifiers")
            timed_transfer = importlib.import_module("omicstl.transfer_forest")
            TIMEDClassifierRF = getattr(timed_classifiers, "TIMEDClassifierRF")
            load_r_functions = getattr(timed_transfer, "load_r_functions")
        except Exception as exc:
            raise ImportError(
                "timed-hpc dependencies are unavailable. Install omicstl + torch + rpy2 and ensure R is configured."
            ) from exc

        load_r_functions()
        try:
            # Preferred integration path: timed-hpc's stable wrapper contract.
            classifier = TIMEDClassifierRF.load(settings.transfer_rf_pkl)
        except Exception:
            # Compatibility fallback for legacy raw TransferForest pickles.
            with open(settings.transfer_rf_pkl, "rb") as pkl_file:
                loaded_obj = pickle.load(pkl_file)
            classifier = _ensure_predictable_transfer_rf(loaded_obj)
        return TimedTransferRFModel(classifier)
        
    print(f"Loading pre-trained model from '{settings.transfer_model_folder}'")
    
    # Import here to avoid circular imports
    from .models import GPRModel, NeuralNetModel, ModelType
    
    if settings.model_type == ModelType.GPR:
        return GPRModel.load_trained_models(settings.transfer_model_folder)
    elif settings.model_type == ModelType.NEURAL_NET:
        return NeuralNetModel.load_trained_models(settings.transfer_model_folder)
    else:
        raise ValueError(f"Unknown model type: {settings.model_type}")


def validate_timed_rf_bridge(model, ingredients_list, ingredients_pd):
    """Run a lightweight preflight check for timed-hpc RF bridge models.

    The bridge model is R-backed and can fail late if feature schema does not
    match what the pickle expects. This check validates the model can score a
    single row with the current ingredient columns.
    """
    if model is None or not hasattr(model, "classifier"):
        return

    row = []
    for ing in ingredients_list:
        ing_rows = ingredients_pd[ingredients_pd["INGREDIENT"] == ing]
        if ing_rows.empty:
            row.append(0.0)
            continue
        nominal = ing_rows["NOMINAL_VALUE"].iloc[0]
        minimum = ing_rows["MIN_VALUE"].iloc[0]
        value = nominal if pd.notna(nominal) else minimum
        if pd.isna(value):
            value = 0.0
        row.append(float(value))

    test_df = pd.DataFrame([row], columns=ingredients_list)
    try:
        model.classifier.predict(test_df)
    except Exception as exc:
        expected_features = _extract_transfer_rf_feature_names(model.classifier)
        details = ""
        if expected_features:
            missing = [f for f in expected_features if f not in ingredients_list]
            extras = [f for f in ingredients_list if f not in expected_features]
            details = (
                f" Expected features: {len(expected_features)}; provided ingredients: {len(ingredients_list)}."
                f" Missing expected features: {len(missing)}; extra provided features: {len(extras)}."
            )
            if missing:
                details += f" Example missing: {missing[:5]}."

        raise RuntimeError(
            "Timed-hpc RF bridge preflight failed. "
            "Check environment (omicstl/rpy2/R) and feature schema compatibility."
            + details
        ) from exc