from dataclasses import dataclass, field
import numpy as np
from .base import _BaseSplineFitter

@dataclass
class SplineFitter(_BaseSplineFitter):
    """
    SplineFitter implementation using C++ extension for performance.
    """
    _cpp_fitter: object = field(init=False, default=None, repr=False)

    def _prepare_matrices(self):
        """
        Initialize the C++ extension fitter.
        """
        from smoothing_spline._spline_extension import SplineFitterCpp as _ExtSplineFitterCpp
        x_scaled, knots_scaled = self._setup_scaling_and_knots()
        self._cpp_fitter = _ExtSplineFitterCpp(x_scaled, knots_scaled, self.w)
        self.N_ = self._cpp_fitter.get_N()
        self.Omega_ = self._cpp_fitter.get_Omega()
        if self.w is not None:
            self.NTW_ = self.N_.T * self.w
        else:
            self.NTW_ = self.N_.T

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-12, 12)):
        """
        Finds the exact lambda value that yields the target degrees of freedom
        using the C++ implementation.
        """
        if target_df >= self.n_k_ - 0.01:
            raise ValueError(f"Target DF ({target_df}) too high.")
        if target_df <= 2.01:
            raise ValueError(f"Target DF ({target_df}) too low.")

        if hasattr(self._cpp_fitter, "solve_for_df"):
             try:
                 # Note: log10_lam_bounds are currently ignored by C++ solver (uses [-12, 12])
                 lam_scaled = self._cpp_fitter.solve_for_df(target_df)
                 return lam_scaled * (self.x_scale_ ** 3)
             except RuntimeError as e:
                 raise RuntimeError(f"C++ solver failed: {e}")
        
        raise RuntimeError("C++ extension required for finding lambda from DF.")

    def solve_gcv(self, y, sample_weight=None, log10_lam_bounds=(-10, 10)):
        """
        Find optimal lambda using GCV and fit the model using C++.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        y_arr = np.asarray(y)
        
        if hasattr(self._cpp_fitter, "solve_gcv"):
             # Adjust bounds for C++ which works on scaled lambda
             shift = 3 * np.log10(self.x_scale_)
             min_log = log10_lam_bounds[0] - shift
             max_log = log10_lam_bounds[1] - shift
             
             try:
                 lam_scaled = self._cpp_fitter.solve_gcv(y_arr, min_log, max_log)
                 self.lamval = lam_scaled * (self.x_scale_ ** 3)
                 self.fit(y)
                 return self.lamval
             except RuntimeError as e:
                 raise RuntimeError(f"C++ GCV solver failed: {e}")
                 
        raise RuntimeError("C++ extension required for GCV optimization.")

    def fit(self, y, sample_weight=None):
        """
        Fit the smoothing spline using the C++ extension.
        """
        self.y = y
        if sample_weight is not None:
            self.w = sample_weight
            if hasattr(self._cpp_fitter, "update_weights"):
                 self._cpp_fitter.update_weights(self.w)
            else:
                 self._prepare_matrices()
        
        if self.lamval is None:
             self.lamval = 0.0
        lam_scaled = self.lamval / self.x_scale_**3
        self.alpha_ = self._cpp_fitter.fit(y, lam_scaled)

        y_hat = self.predict(self.x)
        if self.w is not None:
            X = np.vander(self.x, 2)
            Xw = X * self.w[:, None]
            yw = y_hat * self.w
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        else:
            beta = np.polyfit(self.x, y_hat, 1)

        self.intercept_ = beta[1]
        self.coef_ = beta[0]

    def predict(self, x):
        """
        Predict the response for a new set of predictor variables using C++ basis evaluation.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        Returns
        -------
        np.ndarray
            The predicted response.
        """
        from smoothing_spline._spline_extension import compute_natural_spline_basis
        x_scaled = (x - self.x_min_) / self.x_scale_
        
        # compute_natural_spline_basis returns the basis matrix N
        # We assume extrapolate_linear=True (default in C++)
        N_new = compute_natural_spline_basis(x_scaled, self.knots_scaled_)
        
        # Prediction is N * alpha
        return N_new @ self.alpha_

    def update_weights(self, w):
        """
        Update the weights and refit the model using C++.
        """
        self.w = w
        if hasattr(self._cpp_fitter, "update_weights"):
             self._cpp_fitter.update_weights(self.w)
        else:
             self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        if hasattr(self, 'y'):
            self.fit(self.y)
