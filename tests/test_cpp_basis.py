
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
try:
    from smoothing_spline._spline_extension import compute_natural_spline_basis
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")
def test_cpp_basis_matches_scipy():
    assert CPP_AVAILABLE
    rng = np.random.default_rng(42)
    # Generate random data
    x = np.sort(rng.uniform(0, 10, 100))
    knots = np.sort(rng.uniform(0, 10, 10)) # 10 knots

    # Scipy Basis
    # N matrix where N[i, j] is basis j evaluated at x[i]
    # We can get this by evaluating splines for each basis vector
    identity = np.eye(len(knots))
    cs = CubicSpline(knots, identity, bc_type='natural')
    
    # cs(x) returns shape (len(x), len(knots))
    expected_N = cs(x)
    
    # C++ Basis
    cpp_N = compute_natural_spline_basis(x, knots)
    
    # Compare
    np.testing.assert_allclose(cpp_N, expected_N, atol=1e-8, rtol=1e-8)

@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extension not built")
def test_cpp_basis_extrapolation():
    # Test extrapolation behavior
    knots = np.array([0., 1., 2.])
    x = np.array([-0.5, 0.5, 1.5, 2.5])
    
    identity = np.eye(len(knots))
    cs = CubicSpline(knots, identity, bc_type='natural')
    expected_N = cs(x)
    
    cpp_N = compute_natural_spline_basis(x, knots)
    
    np.testing.assert_allclose(cpp_N, expected_N, atol=1e-8)

