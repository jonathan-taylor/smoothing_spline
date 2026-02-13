# Project Overview

This project is a Python package named `smoothing_spline` that provides a minimal smoothing spline fitter. It includes a `SplineFitter` class for fitting smoothing splines to data, a common technique in non-parametric regression. The implementation is designed to be compatible with the scikit-learn ecosystem.

The core logic is implemented in C++ using `Eigen` and `pybind11` for high performance, with a Python fallback available. The C++ extension (`_spline_extension`) provides:
*   Efficient basis matrix ($N$) construction with optional linear extrapolation.
*   Sparse penalty matrix ($\Omega$) construction.
*   Two solvers:
    *   **Basis Form:** $O(NK^2)$ general solver (allows $K < N$).
    *   **Reinsch Form:** $O(N)$ solver (specifically for $K=N$).
*   Degrees of Freedom ($df$) calculation via trace estimation.
*   Generalized Cross-Validation (GCV) scoring.

# Building and Running

## Installation

To install the package for development, run the following command from the project root. This requires a C++ compiler, `cmake` (optional but recommended), and `pybind11` (installed automatically via `pyproject.toml`).

```bash
pip install -e .
```

## Running Tests

To run the tests, first install the development dependencies:

```bash
pip install -e ".[dev]"
```

Then, run `pytest` from the project root:

```bash
pytest
```

# Development Conventions

The project uses `setuptools` and `setuptools_scm` for package building and version management. The dependencies are listed in `pyproject.toml`.

## C++ Extension
The C++ code is located in `src/`. `src/spline_basis.cpp` contains the implementation using Eigen. The `Eigen` library is included as a git submodule in `src/eigen`.

## Theory
See `docs/theory.md` for a detailed explanation of the mathematical formulation, implementation details (Reinsch vs Basis form), and comparison with R's `smooth.spline`.
