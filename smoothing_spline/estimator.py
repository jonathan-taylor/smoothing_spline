# from sklearn.base import BaseEstimator
# from .smoothing_spline import SplineFitter

# @dataclass
# class SmoothingSpline(BaseEstimator):
#     """
#     Penalized natural spline estimator.
#     This estimator fits a smoothing spline to the data, a popular method for
#     non-parametric regression. The smoothness of the spline is controlled by
#     a penalty parameter, which can be chosen automatically by specifying the
#     desired degrees of freedom.
#     Parameters
#     ----------
#     lamval : float, optional
#         The penalty parameter `lambda`. It is generally recommended to specify
#         the degrees of freedom `df` instead of `lamval`. If neither is
#         provided, `lamval` defaults to 0, which corresponds to interpolation.
#     df : int, optional
#         The desired degrees of freedom. This is used to automatically
#         determine the penalty parameter `lamval`.
#     degrees_of_freedom : int, optional
#         An alias for `df`.
#     knots : np.ndarray, optional
#         The interior knots to use for the spline. If not specified, the unique
#         values of `x` are used.
#     n_knots : int, optional
#         The number of knots to use. If `knots` is not specified, `n_knots`
#         are chosen uniformly from the percentiles of `x`.
#     Attributes
#     ----------
#     fitter_ : SplineFitter
#         The internal fitter object that contains the detailed results of the
#         fit, including the fitted spline, coefficients, and linear components.
#     """
#     lamval: float = None
#     df: int = None
#     degrees_of_freedom: int = None
#     knots: np.ndarray = None
#     n_knots: int = None
#     fitter_: SplineFitter = field(init=False, repr=False)

#     def __post_init__(self):
#         if self.degrees_of_freedom is not None:
#             if self.df is not None:
#                 raise ValueError("Only one of `df` or `degrees_of_freedom` should be provided.")
#             self.df = self.degrees_of_freedom
        
#         if self.lamval is None and self.df is None:
#             self.lamval = 0
        
#         if self.lamval is not None and self.df is not None:
#             raise ValueError("Only one of `lamval` or `df` can be provided.")

#     def fit(self, x, y, w=None):
#         """
#         Fit the smoothing spline to the data.
#         Parameters
#         ----------
#         x : np.ndarray
#             The predictor variable.
#         y : np.ndarray
#             The response variable.
#         w : np.ndarray, optional
#             Weights for the observations. Defaults to None.
#         Returns
#         -------
#         self : SmoothingSpline
#             The fitted estimator.
#         """
#         self.fitter_ = SplineFitter(x,
#                                      w=w,
#                                      lamval=self.lamval,
#                                      df=self.df,
#                                      knots=self.knots,
#                                      n_knots=self.n_knots)
#         self.fitter_.fit(y)
#         return self

#     def predict(self, x):
#         """
#         Predict the response for a new set of predictor variables.
#         Parameters
#         ----------
#         x : np.ndarray
#             The predictor variables.
#         Returns
#         -------
#         np.ndarray
#             The predicted response.
#         """
#         return self.fitter_.predict(x)

# def compute_edf_reinsch(x, lamval, weights=None):
#     x = np.array(x, dtype=float)
#     n = len(x)
#     if weights is None:
#         w_inv_vec = np.ones(n)
#     else:
#         weights = np.array(weights, dtype=float)
#         w_inv_vec = 1.0 / (weights + 1e-12)
#     W_inv = sparse.diags(w_inv_vec)
#     h = np.diff(x)
#     inv_h = 1.0 / h
#     main_diag_R = (h[:-1] + h[1:]) / 3.0
#     off_diag_R = h[1:-1] / 6.0
#     R = sparse.diags([off_diag_R, main_diag_R, off_diag_R],
#                      [-1, 0, 1], shape=(n-2, n-2))
#     d0 = inv_h[:-1]
#     d1 = -inv_h[:-1] - inv_h[1:]
#     d2 = inv_h[1:]
#     Q = sparse.diags([d0, d1, d2], [0, -1, -2], shape=(n, n-2))
#     B = Q.T @ W_inv @ Q
#     M_trace = R + lamval * B
#     M_trace = M_trace.tocsc()
#     B = B.tocsc()
#     X = splinalg.spsolve(M_trace, B)
#     if sparse.issparse(X):
#         tr_val = X.diagonal().sum()
#     else:
#         tr_val = np.trace(X)
#     return n - lamval * tr_val

# def estimate_edf_hutchinson(x, lamval, weights=None, n_vectors=50):
#     x = np.array(x, dtype=float)
#     n = len(x)
#     if weights is None: w_inv_vec = np.ones(n)
#     else: w_inv_vec = 1.0 / (np.array(weights, dtype=float) + 1e-12)
#     W_inv = sparse.diags(w_inv_vec)
#     h = np.diff(x)
#     inv_h = 1.0 / h
#     R = sparse.diags([h[1:-1]/6.0, (h[:-1]+h[1:])/3.0, h[1:-1]/6.0], [-1, 0, 1], shape=(n-2, n-2))
#     Q = sparse.diags([inv_h[:-1], -inv_h[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n, n-2))
#     B = Q.T @ W_inv @ Q
#     M_trace = R + lamval * B
#     M_trace = M_trace.tocsc()
#     solve_M = splinalg.factorized(M_trace)
#     trace_est = 0.0
#     dim = n - 2
#     for i in range(n_vectors):
#         z = np.random.choice([-1, 1], size=dim)
#         v = B @ z
#         u = solve_M(v)
#         trace_est += np.dot(z, u)
#     mean_trace = trace_est / n_vectors
#     return n - lamval * mean_trace

