from abc import ABC, abstractmethod
from enum import Enum
import os

import numpy as np
import torch

# from constants import *
from ..ml import neural_networks as net
from ..ml import gaussian_process as gpr
from ..ml import random_forest as rf

class ModelType(Enum):
    GPR = 0
    NEURAL_NET = 1
    RF = 2


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


class RandomForestModel(Model):
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.is_trained = False

        # Use the enum name defined in your project
        super().__init__(None, ModelType.RF)

    @classmethod
    def load_trained_models(cls, models_path):
        """
        Load a trained Random Forest model from models_path.
        """

        model_file = os.path.join(
            models_path,
            "random_forest_model.joblib"
        )

        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Random Forest model not found: {model_file}"
            )

        obj = cls(models_path)
        obj.model = joblib.load(model_file)
        obj.is_trained = True

        return obj

    def check_path(self):
        """
        Create the model directory if it does not exist.
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

    def train(self, X_train, y_train, **kwargs):
        """
        Train and save the Random Forest model.
        """

        self.check_path()

        self.model = rf.train_new_RF(
            X_train,
            y_train,
            self.model_path,
            **kwargs
        )

        self.is_trained = True

        return self.model

    def evaluate(self, X, clip=True, n=1):
        """
        Generate Random Forest predictions.

        Parameters
        ----------
        X : array-like
            Input features.

        clip : bool
            Clip predictions to the range [0, 1].

        n : int
            Number of approximate predictive samples to generate.

        Returns
        -------
        samples : numpy.ndarray
            Predictions generated from individual trees.

        variances : numpy.ndarray
            Variance of tree predictions for each input row.
        """

        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "Random Forest model needs to be trained "
                "or loaded before evaluating."
            )

        samples, variances = rf.sample_RF(
            model=self.model,
            X=X,
            n_samples=n
        )

        if clip:
            samples = np.clip(samples, 0, 1)

        return samples, variances