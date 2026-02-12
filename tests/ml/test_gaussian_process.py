"""Tests for Gaussian process utilities."""
import numpy as np

from src.bacterai.ml.gaussian_process import make_positive_semidefinite


def test_make_positive_semidefinite_clips_negatives():
    mat = np.array([[2.0, -3.0], [-3.0, 2.0]])
    fixed = make_positive_semidefinite(mat)
    eigvals = np.linalg.eigvalsh(fixed)

    assert np.all(eigvals >= 0)
