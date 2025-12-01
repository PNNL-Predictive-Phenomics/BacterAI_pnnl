"""Transfer learning utilities for BacterAI experiments."""

import os
import shutil
import pandas as pd
from ..analysis import processing as utils
from ..scripts.size_n_to_m_conversion import fill_new_ingredients
from ..utils.constants import AA_SHORT, BASE_NAMES


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
    if settings.transfer_model_folder is None:
        return None
        
    print(f"Loading pre-trained model from '{settings.transfer_model_folder}'")
    
    # Import here to avoid circular imports
    from .models import GPRModel, NeuralNetModel, ModelType
    
    if settings.model_type == ModelType.GPR:
        return GPRModel.load_trained_models(settings.transfer_model_folder)
    elif settings.model_type == ModelType.NEURAL_NET:
        return NeuralNetModel.load_trained_models(settings.transfer_model_folder)
    else:
        raise ValueError(f"Unknown model type: {settings.model_type}")