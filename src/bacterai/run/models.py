from abc import ABC, abstractmethod
from enum import Enum
import os

import numpy as np
import pandas as pd

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
    """Adapter for timed-hpc TransferForest pickle models."""

    def __init__(self, classifier, feature_names=None):
        self.classifier = classifier
        self.feature_names = feature_names or []
        super().__init__(None, ModelType.NEURAL_NET)

    def set_feature_names(self, feature_names):
        self.feature_names = list(feature_names)

    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError("TimedTransferRFModel is inference-only in this bridge mode.")

    def evaluate(self, X, clip=True):
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = pd.DataFrame(X)

        predictions = np.asarray(self.classifier.predict(X_df), dtype=float)

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