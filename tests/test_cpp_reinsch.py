import numpy as np
import pytest
from smoothing_spline.fitter import SplineFitter

from smoothing_spline._spline_extension import ReinschFitter

def test_reinsch_fitter_exact_fit():
    rng = np.random.default_rng(200)
    x = np.sort(rng.uniform(0, 10, 50))
    # Reinsch is for n_knots == n_x
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    # Python SplineFitter (Basis form)
    py_fitter = SplineFitter(x, knots=x, engine='natural')
    py_fitter.fit(y)
    
    # Reinsch Fitter
    # Pass SCALED x to match physics if we want to use same lambda scaling
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschFitter(x_scaled, None)
    
    lamval = 0.5
    lam_scaled = lamval / scale**3
    
    # Basis fit
    py_fitter.lamval = lamval # This will use basis form inside
    py_fitter.fit(y)
    basis_pred = py_fitter.predict(x)
    
    # Reinsch fit
    # fit returns fitted values directly
    reinsch_pred = reinsch_fitter.fit(y, lam_scaled)
    
    np.testing.assert_allclose(reinsch_pred, basis_pred, atol=1e-5)

def test_reinsch_df():
    rng = np.random.default_rng(201)
    x = np.sort(rng.uniform(0, 10, 50))
    
    py_fitter = SplineFitter(x, knots=x, engine='natural')
    py_fitter._prepare_matrices()
    
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschFitter(x_scaled, None)
    
    lamval = 0.1
    lam_scaled = lamval / scale**3
    
    # Calculate DF via Basis (slow)
    # df = trace(S)
    basis_df = py_fitter._cpp_fitter.compute_df(lam_scaled)
    
    # Calculate DF via Reinsch (fast)
    reinsch_df = reinsch_fitter.compute_df(lam_scaled)
    
    np.testing.assert_allclose(reinsch_df, basis_df, atol=1e-5)

def test_reinsch_gcv():
    rng = np.random.default_rng(202)
    x = np.sort(rng.uniform(0, 10, 50))
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    py_fitter = SplineFitter(x, knots=x, engine='natural')
    py_fitter.fit(y)
    
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschFitter(x_scaled, None)
    
    lamval = 0.1
    lam_scaled = lamval / scale**3
    
    # Basis GCV
    basis_gcv = py_fitter._cpp_fitter.gcv_score(lam_scaled, y)
    
    # Reinsch GCV
    reinsch_gcv = reinsch_fitter.gcv_score(lam_scaled, y)
    
    np.testing.assert_allclose(reinsch_gcv, basis_gcv, atol=1e-5)
