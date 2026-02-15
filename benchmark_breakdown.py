import time
import numpy as np
from smoothing_spline.fitter import SplineFitter

def benchmark(n):
    print(f"Benchmark N={n}")
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
    
    fitter = SplineFitter(x)
    # Force initialization
    fitter._prepare_matrices()
    
    # Access internal C++ fitter
    cpp = fitter._cpp_fitter
    lam = 0.001
    
    # Measure fit
    start = time.time()
    for _ in range(10):
        cpp.fit(y, lam)
    end = time.time()
    fit_time = (end - start) / 10.0
    print(f"Fit time (avg of 10): {fit_time:.6f} s")
    
    # Measure compute_df
    # We need to pass the correctly scaled lambda if we were calling from Python wrapper,
    # but calling internal cpp method directly expects whatever the internal logic expects.
    # In SplineFitterReinschCpp, compute_df takes raw lambda? No, it takes the lambda passed to fit.
    # Let's just use a value.
    start = time.time()
    n_df_runs = 5 if n < 2000 else 1
    for _ in range(n_df_runs):
        cpp.compute_df(lam)
    end = time.time()
    df_time = (end - start) / n_df_runs
    print(f"Compute DF time (avg of {n_df_runs}): {df_time:.6f} s")
    
    print(f"Ratio (DF / Fit): {df_time / fit_time:.2f}x")

if __name__ == "__main__":
    for n in [500, 1000, 5000]:
        benchmark(n)
