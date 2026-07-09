from abc import ABC, abstractmethod
from enum import Enum
import os
import pickle

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

try:
    import torch
except Exception:
    torch = None

# from constants import *
try:
    from ..ml import neural_networks as net
except Exception:
    net = None

try:
    from ..ml import gaussian_process as gpr
except Exception:
    gpr = None


class ModelType(Enum):
    GPR = 0
    NEURAL_NET = 1
    TRANSFER_RF = 2


class Model(ABC):
    def __init__(self, model, model_type):
        self.model_type = model_type

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def close(self):
        pass

    def get_type(self):
        return self.model_type

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X):
        pass


class GPRModel(Model):
    def __init__(self, model_path):
        # self.activate_R()
        self.model = []
        self.likelihood = []
        self.model_path = model_path
        self.is_trained = False
        super().__init__(None, ModelType.GPR)
        
    @classmethod
    def load_trained_models(cls, models_path):
        if torch is None or net is None:
            raise ImportError("PyTorch dependencies are required to load GPR models.")
        obj = cls(models_path)

        for filename in os.listdir(models_path):
            if "model" in filename:
                model = torch.load(os.path.join(models_path, filename), map_location=torch.device(net.DEVICE))
                obj.model.append(model)
            if "likelihood" in filename:
                likelihood = torch.load(os.path.join(models_path, filename), map_location=torch.device(net.DEVICE))
                obj.likelihood.append(likelihood)

        obj.is_trained = True
        return obj
    
    def check_path(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def train(self, X_train, y_train, **kwargs):
        if gpr is None:
            raise ImportError("GPR dependencies are unavailable. Install required ML packages.")
        # X_trainR = robjects.r.matrix(
        #     X_train, nrow=X_train.shape[0], ncol=X_train.shape[1]
        # )
        # y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
        # self.model = self.gpr_lib.train_new_GP(X_trainR, y_trainR)
        self.check_path()
        self.model, self.likelihood = gpr.train_new_GP(X_train, y_train, self.model_path, **kwargs)
        self.is_trained = True

    def evaluate(self, X, clip=True, n=1):
        # X_evalR = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        if not self.is_trained:
            raise Exception("GPR model needs to be trained before evaluating.")

        samples, variances  = gpr.sample_GP(self.model, self.likelihood, X, n)
        # Do we want to clip samples?
        if clip:
            samples = np.clip(samples, 0, 1)
        return samples, variances

    # def activate_R(self):
    #     with open("gpr_lib.R", "r") as f:
    #         s = f.read()
    #         self.gpr_lib = STAP(s, "gpr_lib")
    #         robjects.r("Sys.setenv(MKL_DEBUG_CPU_TYPE = '5')")
    #     rpyn.activate()

    # def close(self):
    #     # Clean up R's GPR model object
    #     self.gpr_lib.delete_GP(self.model)


class NeuralNetModel(Model):
    def __init__(self, models_path):
        self.models_path = models_path
        self.models = []
        self.is_trained = False
        super().__init__(None, ModelType.NEURAL_NET)

    @classmethod
    def load_trained_models(cls, models_path):
        if torch is None or net is None:
            raise ImportError("PyTorch dependencies are required to load neural network models.")
        obj = cls(models_path)

        for filename in os.listdir(models_path):
            if "bag_model" in filename:
                model = torch.load(os.path.join(models_path, filename), map_location=torch.device(net.DEVICE))
                obj.models.append(model)

        obj.is_trained = True
        return obj

    def check_path(self):
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train(self, X_train, y_train, **kwargs):
        if net is None:
            raise ImportError("Neural network dependencies are unavailable. Install required ML packages.")
        self.check_path()
        self.models = net.train_bagged(X_train, y_train, self.models_path, **kwargs)
        self.is_trained = True

    def evaluate(self, X, clip=True):
        if not self.is_trained:
            raise Exception("Neural net model needs to be trained before evaluating.")

        predictions, variances = net.eval_bagged(X, self.models)
        if clip:
            predictions = np.clip(predictions, 0, 1)
        return predictions, variances


class TimedTransferRFModel(Model):
    """Adapter for persisted Round 1 transfer RF artifacts."""

    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names or []
        super().__init__(None, ModelType.TRANSFER_RF)

    def set_feature_names(self, feature_names):
        self.feature_names = list(feature_names)

    @classmethod
    def load_trained_model(cls, model_path):
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        classifier = payload["classifier"]
        feature_names = payload.get("feature_names", [])
        return cls(classifier=classifier, feature_names=feature_names)

    def save_trained_model(self, model_path):
        payload = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError("TimedTransferRFModel is inference-only in this bridge mode.")

    @staticmethod
    def _predict_from_transfer_forest(classifier, X_df):
        prediction_dict = classifier.generate_predictions([X_df])[0]
        preferred_keys = [
            "pred_ensemble_full",
            "pred_ensemble",
            "pred_0_full",
            "pred_0",
            "pred_1_full",
            "pred_1",
            "pred_source_full",
            "pred_source",
        ]

        for key in preferred_keys:
            if key in prediction_dict:
                return np.asarray(prediction_dict[key], dtype=float)

        for key, value in prediction_dict.items():
            if "_prob" in key:
                continue
            return np.asarray(value, dtype=float)

        raise ValueError("TransferForest returned no usable prediction outputs.")

    def evaluate(self, X, clip=True):
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = pd.DataFrame(X)

        if hasattr(self.classifier, "predict"):
            predictions = np.asarray(self.classifier.predict(X_df), dtype=float)
        elif hasattr(self.classifier, "generate_predictions"):
            predictions = self._predict_from_transfer_forest(self.classifier, X_df)
        else:
            raise AttributeError(
                "Timed transfer classifier must expose either predict() or generate_predictions()."
            )

        # BacterAI assumes a growth score in [0, 1] for simulation thresholds.
        if getattr(self.classifier, "_is_classification", False):
            unique_vals = np.unique(predictions)
            if unique_vals.size > 1:
                predictions = (predictions == unique_vals.max()).astype(float)
            else:
                predictions = np.zeros_like(predictions)

        if clip:
            predictions = np.clip(predictions, 0, 1)

        variances = np.zeros_like(predictions, dtype=float)
        return predictions, variances


class IterativeTransferRFModel(Model):
    """RF model used for iterative transfer-learning simulation rounds."""

    def __init__(self, feature_names=None, input_feature_names=None):
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.input_feature_names = list(input_feature_names) if input_feature_names is not None else []
        self.classifier = None
        self.is_trained = False
        super().__init__(None, ModelType.TRANSFER_RF)

    def set_feature_names(self, feature_names):
        self.feature_names = list(feature_names)

    def set_input_feature_names(self, feature_names):
        self.input_feature_names = list(feature_names)

    @classmethod
    def from_classifier(cls, classifier, feature_names=None, input_feature_names=None):
        obj = cls(feature_names=feature_names, input_feature_names=input_feature_names)
        obj.classifier = classifier
        obj.is_trained = True
        return obj

    @classmethod
    def load_trained_model(cls, model_path):
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            feature_names=payload.get("feature_names", []),
            input_feature_names=payload.get("input_feature_names", []),
        )
        obj.classifier = payload["classifier"]
        obj.is_trained = True
        return obj

    def save_trained_model(self, model_path):
        if not self.is_trained or self.classifier is None:
            raise ValueError("Cannot save IterativeTransferRFModel before training.")
        payload = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "input_feature_names": self.input_feature_names,
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

    def train(self, X_train, y_train, **kwargs):
        if RandomForestRegressor is None:
            raise ImportError("scikit-learn is required for iterative transfer RF training.")
        if X_train is None or y_train is None or len(X_train) == 0:
            raise ValueError("IterativeTransferRFModel requires non-empty training data.")

        n_estimators = int(kwargs.get("n_estimators", 400))
        random_state = int(kwargs.get("random_state", 42))
        min_samples_leaf = int(kwargs.get("min_samples_leaf", 1))

        self.classifier = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=min_samples_leaf,
        )
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

    def evaluate(self, X, clip=True):
        if not self.is_trained or self.classifier is None:
            raise Exception("Iterative transfer RF model needs to be trained before evaluating.")

        if isinstance(X, pd.DataFrame):
            X_eval = X.copy()
        elif self.input_feature_names and len(self.input_feature_names) == X.shape[1]:
            X_eval = pd.DataFrame(X, columns=self.input_feature_names)
        elif self.feature_names and len(self.feature_names) == X.shape[1]:
            X_eval = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_eval = pd.DataFrame(X)

        X_pred = X_eval
        if self.feature_names and isinstance(X_eval, pd.DataFrame):
            for col in self.feature_names:
                if col not in X_eval.columns:
                    X_eval[col] = 0.0
            X_pred = X_eval[self.feature_names]

        if hasattr(self.classifier, "predict"):
            predictions = np.asarray(self.classifier.predict(X_pred), dtype=float)
        elif hasattr(self.classifier, "generate_predictions"):
            if isinstance(X_pred, pd.DataFrame):
                X_df = X_pred
            else:
                X_df = pd.DataFrame(X_pred)
            predictions = TimedTransferRFModel._predict_from_transfer_forest(self.classifier, X_df)
        else:
            raise AttributeError(
                "Iterative transfer classifier must expose either predict() or generate_predictions()."
            )

        # Use tree-level spread as a predictive uncertainty proxy when available.
        if hasattr(self.classifier, "estimators_") and self.classifier.estimators_:
            tree_predictions = np.asarray(
                [estimator.predict(X_eval) for estimator in self.classifier.estimators_],
                dtype=float,
            )
            variances = np.var(tree_predictions, axis=0)
        else:
            variances = np.zeros_like(predictions, dtype=float)

        if clip:
            predictions = np.clip(predictions, 0, 1)

        return predictions, variances