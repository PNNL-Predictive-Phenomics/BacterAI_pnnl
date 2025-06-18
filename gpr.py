import os
import torch
import gpytorch
import numpy as np

from sklearn.metrics import mean_squared_error  # Importing mean_squared_error
from scipy.stats import multivariate_normal

# Code built from generative AI, based on operations and parameters from gpr_lib.R

# Define a simple GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_new_GP(X, y, model_path, d=0.1, g=0.1, max_iter=100, lr=0.1, verbosity=2):
    # Convert data to tensors
    train_x = torch.tensor(X, dtype=torch.float32)
    train_y = torch.tensor(y, dtype=torch.float32)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=lr)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(max_iter):
        optimizer.zero_grad()  # Zero gradients from previous iteration
        output = model(train_x)
        loss = -mll(output, train_y)  # Calculate loss
        loss.backward()  # Backprop gradients
        optimizer.step()  # Update model parameters
        model.train()
        likelihood.train()
        
        if verbosity > 1:
            print(f'Iter {i + 1}/{max_iter} | Train loss: {loss.item():.4f}')
    
    # Final MSE calculation after training completes
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
        model.eval()
        likelihood.eval()
        y_pred = model(train_x).mean
        final_mse = mean_squared_error(train_y.numpy(), y_pred.numpy())
    print(f'Final MSE: {final_mse:.4f}')
    
    torch.save(model.state_dict(), os.path.join(model_path, "gpr_model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(model_path, "gpr_likelihood.pth"))
    
    return model, likelihood


def sample_GP(model, likelihood, X, n_samples=1):
    # Convert data to tensor
    test_x = torch.tensor(X, dtype=torch.float32)
    
    # Set into evaluation mode
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    
    # Get the mean and covariance
    mean = observed_pred.mean.numpy()
    cov = observed_pred.covariance_matrix.numpy()
    
    # Sample from the multivariate normal distribution
    samples = np.atleast_1d(multivariate_normal.rvs(mean, cov, size=n_samples))
    variances = np.diag(cov)
    
    return samples, variances

