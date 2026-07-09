"""Transfer learning utilities for BacterAI experiments."""

import os
import shutil
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..analysis import processing as utils
from ..ml.transfer_forest import TransferForestPython
from ..scripts.size_n_to_m_conversion import fill_new_ingredients
from ..utils.constants import AA_SHORT, BASE_NAMES
from .recommendation import recommend_next_batch


PPUTIDA_FEATURES = [
    "d_glucose",
    "sodium_citrate",
    "sodium_octanoate",
    "sodium_acetate",
    "sodium_benzoate",
    "d_xylose",
    "l_arabinose",
    "sodium_chloride",
    "urea",
    "ammonium_chloride",
    "pH",
]


def _resolve_transfer_rf_data_dir() -> pathlib.Path:
    this_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        this_dir.parent / "data" / "transfer_rf",
        pathlib.Path.cwd() / "src" / "bacterai" / "data" / "transfer_rf",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError("Could not locate bundled transfer RF data directory: src/bacterai/data/transfer_rf")


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _build_feature_specs(ingredients_pd, ingredients_list):
    selected_features = [col for col in PPUTIDA_FEATURES if col in ingredients_list]
    if len(selected_features) < 2:
        raise ValueError(
            "Transfer-learning mode requires at least 2 overlapping features with bundled pputida RF features. "
            f"Found overlap: {selected_features}"
        )

    feature_ranges = {}
    step_sizes = {}
    defaults = {}

    for ing in ingredients_list:
        ing_rows = ingredients_pd[ingredients_pd["INGREDIENT"] == ing]
        if ing_rows.empty:
            defaults[ing] = 0.0
            continue

        row = ing_rows.iloc[0]
        min_val = _safe_float(row.get("MIN_VALUE"), default=0.0)
        max_val = _safe_float(row.get("MAX_VALUE"), default=min_val)
        nominal = _safe_float(row.get("NOMINAL_VALUE"), default=min_val)

        if nominal < min_val:
            nominal = min_val
        if nominal > max_val:
            nominal = max_val
        defaults[ing] = nominal

        if ing not in selected_features:
            continue

        feature_ranges[ing] = (min_val, max_val)
        ing_type = str(row.get("TYPE", "")).strip().lower()
        n_states = _safe_int(row.get("N_STATES"), default=0)
        if ing_type == "semi-quantitative" and n_states > 1 and max_val > min_val:
            step_sizes[ing] = (max_val - min_val) / (n_states - 1)

    return selected_features, feature_ranges, step_sizes, defaults


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def create_transfer_learning_round1_batch(settings, ingredients_pd, ingredients_list):
    """Train native Python transfer RF from bundled source/target data and generate Round 1 batch."""
    if settings.round_number != 1:
        raise RuntimeError("Transfer-learning mode currently supports Round 1 only.")

    data_dir = _resolve_transfer_rf_data_dir()

    selected_features, feature_ranges, step_sizes, defaults = _build_feature_specs(
        ingredients_pd,
        ingredients_list,
    )

    target_raw = pd.read_csv(data_dir / "pputida_target.csv")
    target_df = target_raw[["fitness"] + selected_features].rename(columns={"fitness": "response"})

    tgt_combo, tgt_test = train_test_split(
        target_df,
        train_size=130 + 18,
        test_size=50,
        random_state=42,
    )
    tgt_train, tgt_ensemble = train_test_split(
        tgt_combo,
        train_size=130,
        test_size=18,
        random_state=42,
    )

    source_files = {
        "P. putida only": "pputida_source_putida.csv",
        "All Pseudomonas": "pputida_source_pseudomonas.csv",
        "All Gammaproteobacteria": "pputida_source_gammaproteobacteria.csv",
    }

    source_evaluations = {}
    best_model = None
    best_source = None
    best_rmse = float("inf")

    for source_label, source_file in source_files.items():
        source_raw = pd.read_csv(data_dir / source_file)
        source_df = source_raw[["Resp"] + selected_features].rename(columns={"Resp": "response"})

        rf_model = TransferForestPython(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=1,
        )
        rf_model.fit(
            source_X=source_df[selected_features],
            source_y=source_df["response"].to_numpy(dtype=float),
            target_X=tgt_train[selected_features],
            target_y=tgt_train["response"].to_numpy(dtype=float),
        )

        pred_dict = rf_model.generate_predictions(
            views=[tgt_test[selected_features]],
            response=tgt_test["response"].to_numpy(dtype=float),
            validation_views=[tgt_test[selected_features]],
            validation_response=tgt_test["response"].to_numpy(dtype=float),
            ensemble_views=[tgt_ensemble[selected_features]],
            ensemble_response=tgt_ensemble["response"].to_numpy(dtype=float),
        )[0]

        pred = np.asarray(pred_dict.get("pred_ensemble", pred_dict.get("pred_0")), dtype=float)
        rmse = _rmse(tgt_test["response"].to_numpy(dtype=float), pred)

        source_evaluations[source_label] = {
            "source_rows": int(len(source_df)),
            "rf_rmse": rmse,
        }
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = rf_model
            best_source = source_label

    if best_model is None:
        raise RuntimeError("Failed to train transfer-learning RF model from bundled datasets")

    feature_cols = selected_features

    # Persist transfer RF artifact so later rounds can warm-start native MDP.
    from .models import TimedTransferRFModel
    timed_model = TimedTransferRFModel(best_model, feature_names=feature_cols)
    round_folder = pathlib.Path(settings.experiment_path) / f"Round{settings.round_number}"
    round_folder.mkdir(parents=True, exist_ok=True)

    timed_model_path = round_folder / "transfer_rf_model.pkl"
    timed_model.save_trained_model(str(timed_model_path))

    batch = recommend_next_batch(
        model=best_model,
        existing_data=tgt_combo[["response"] + feature_cols],
        response_col="response",
        feature_cols=feature_cols,
        feature_ranges=feature_ranges,
        step_sizes=step_sizes,
        batch_size=settings.batch_size,
        acquisition="EI",
        n_candidates=8192,
        n_mc_samples=50,
        shortlist_pct=0.05,
        seed=42,
        return_candidates=False,
    )

    for ing in ingredients_list:
        if ing not in batch.columns:
            batch[ing] = defaults.get(ing, 0.0)

    ordered_cols = ingredients_list + [
        col
        for col in ["predicted_mean", "predicted_std", "acquisition_score", "batch_rank"]
        if col in batch.columns
    ]
    batch = batch[ordered_cols].copy()

    predicted_mean = batch["predicted_mean"] if "predicted_mean" in batch.columns else pd.Series(1.0, index=batch.index)
    predicted_std = batch["predicted_std"] if "predicted_std" in batch.columns else pd.Series(0.0, index=batch.index)

    batch["type"] = "transfer_rf"
    batch["direction"] = 2
    batch["frontier_type"] = "FRONTIER"
    batch["growth_pred"] = predicted_mean.astype(float).clip(lower=0.0, upper=1.0)
    batch["var"] = (predicted_std.astype(float) ** 2)
    batch["is_redo"] = False
    batch["round"] = 1

    metrics = {
        "mode": "transfer_learning_rf",
        "selected_source": best_source,
        "selected_source_rmse": float(best_rmse),
        "source_evaluations": source_evaluations,
        "n_recommendations": int(len(batch)),
        "feature_columns": feature_cols,
        "transfer_rf_model_artifact": str(timed_model_path),
        "implementation": "python_only_transfer_forest",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    return batch, {"TRANSFER_RF": metrics}


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
    # Import here to avoid circular imports
    from .models import GPRModel, NeuralNetModel, TimedTransferRFModel, ModelType

    if settings.model_type == ModelType.TRANSFER_RF and settings.transfer_learning and settings.round_number > 1:
        transfer_model_path = os.path.join(
            settings.experiment_path,
            "Round1",
            "transfer_rf_model.pkl",
        )
        if not os.path.exists(transfer_model_path):
            raise FileNotFoundError(
                "Transfer RF warm-start artifact not found at "
                f"'{transfer_model_path}'. Run Round 1 with --transfer-learning first."
            )
        print(f"Loading transfer RF model from '{transfer_model_path}'")
        return TimedTransferRFModel.load_trained_model(transfer_model_path)

    if settings.transfer_model_folder is None:
        return None
        
    print(f"Loading pre-trained model from '{settings.transfer_model_folder}'")
    
    if settings.model_type == ModelType.GPR:
        return GPRModel.load_trained_models(settings.transfer_model_folder)
    elif settings.model_type == ModelType.NEURAL_NET:
        return NeuralNetModel.load_trained_models(settings.transfer_model_folder)
    else:
        raise ValueError(f"Unknown model type: {settings.model_type}")


def validate_timed_rf_bridge(model, ingredients_list, ingredients_pd):
    """Legacy no-op retained for backward compatibility."""
    if model is None:
        return