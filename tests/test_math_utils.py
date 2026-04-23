import numpy as np
import pytest
from utils.math_utils import skew


def test_skew_equals_cross_product():
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_skew_is_antisymmetric():
    v = np.array([1.0, 2.0, 3.0])
    S = skew(v)
    np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-14)


def test_skew_zero_diagonal():
    v = np.array([5.0, -3.0, 0.5])
    S = skew(v)
    np.testing.assert_allclose(np.diag(S), np.zeros(3), atol=1e-14)


def test_skew_output_shape():
    assert skew(np.zeros(3)).shape == (3, 3)
