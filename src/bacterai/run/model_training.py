# core/model_training.py
"""Model training utilities for BacterAI experiments."""

import os
import shutil
from .models import GPRModel, NeuralNetModel, IterativeTransferRFModel, TimedTransferRFModel, ModelType


def train_experiment_model(X_train, y_train, settings, new_round_folder, transfer_model):
    """
    Train a model based on configuration and training data.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training input data
    y_train : np.ndarray  
        Training target data
    settings : Settings
        Experiment settings object
    new_round_folder : Path
        Path to new round folder
    transfer_model : Model or None
        Pre-trained model for transfer learning
        
    Returns
    -------
    Model
        Trained model ready for use
    """
    n_ingredients = len(X_train[0]) if X_train is not None and len(X_train) > 0 else None

    if settings.model_type == ModelType.TRANSFER_RF:
        if (
            settings.transfer_learning
            and settings.round_number == 2
            and isinstance(transfer_model, TimedTransferRFModel)
        ):
            print("Warm-starting Round 2 TRANSFER_RF from Round 1 transfer artifact...")
            model = IterativeTransferRFModel.from_classifier(
                transfer_model.classifier,
                feature_names=transfer_model.feature_names,
                input_feature_names=getattr(settings, "ingredient_names", None),
            )
        else:
            print("Training iterative Transfer RF model...")
            model = IterativeTransferRFModel()
            model.train(
                X_train,
                y_train,
                n_estimators=max(int(settings.n_bags) * 16, 200),
                random_state=42,
                min_samples_leaf=1,
            )
        artifact_path = os.path.join(new_round_folder, "transfer_rf_model.pkl")
        model.save_trained_model(artifact_path)
        print(f"Saved iterative Transfer RF artifact to {artifact_path}")
        return model
    
    if settings.model_type == ModelType.GPR:
        # Train GPR Model
        print("Training GPR model...")
        models_folder = os.path.join(new_round_folder, "gpr_model")
        model = GPRModel(models_folder)
        model.train(X_train, y_train)
        return model

    elif settings.transfer_model_folder and settings.round_number == 1:
        # Use purely pre-trained NN model for 1st round
        print(f"Using pre-trained NN model from {settings.transfer_model_folder}")
        models_folder = os.path.join(new_round_folder, "nn_models")
        if os.path.exists(models_folder):
            raise Exception(
                f"File exists: '{models_folder}'. Cannot copy pre-trained models "
                f"to here unless you remove it first."
            )
        shutil.copytree(settings.transfer_model_folder, models_folder)
        return transfer_model
        
    else:
        # Train new NN model (possibly with transfer learning)
        print("Training new Neural Network model...")
        models_folder = os.path.join(new_round_folder, "nn_models")
        model = NeuralNetModel(models_folder)
        transfer_models = transfer_model.models if settings.transfer_model_folder else []
        
        model.train(
            X_train,
            y_train,
            n_ingredients=n_ingredients,
            n_bags=settings.n_bags,
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
            transfer_models=transfer_models,
        )
        return model