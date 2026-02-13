#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace py = pybind11;

/**
 * Computes the Natural Cubic Spline Basis matrix.
 * 
 * This function replicates the behavior of:
 * scipy.interpolate.CubicSpline(knots, np.eye(len(knots)), bc_type='natural')(x)
 * 
 * @param x Vector of evaluation points.
 * @param knots Vector of knot locations (must be sorted).
 * @return Matrix N where N(i, j) is the value of the j-th basis spline at x[i].
 */
Eigen::MatrixXd compute_natural_spline_basis(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& knots
) {
    long n_knots = knots.size();
    long n_x = x.size();

    // 1. Setup the tridiagonal system for second derivatives (M)
    // The system is A * M = rhs
    // A is (n_knots-2) x (n_knots-2) tridiagonal matrix
    // rhs depends on the y values (which are columns of Identity matrix)

    Eigen::VectorXd h = knots.segment(1, n_knots - 1) - knots.segment(0, n_knots - 1);
    
    // Matrix A (symmetric tridiagonal)
    // Dimensions: (n_knots - 2) x (n_knots - 2)
    // Diagonal: (h[i] + h[i+1]) / 3
    // Off-diagonal: h[i+1] / 6
    
    long n_inner = n_knots - 2;
    if (n_inner < 0) {
        // Handle small number of knots (linear or constant)
        // For < 2 knots, behavior is degenerate, maybe just return something or throw
        if (n_knots == 2) {
             // Linear interpolation
             Eigen::MatrixXd N(n_x, 2);
             for (long i = 0; i < n_x; ++i) {
                 double t = (x[i] - knots[0]) / h[0];
                 // Linear extrapolation
                 N(i, 0) = 1.0 - t;
                 N(i, 1) = t;
             }
             return N;
        }
        throw std::runtime_error("Need at least 2 knots.");
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_inner, n_inner);
    
    for (long i = 0; i < n_inner; ++i) {
        A(i, i) = (h[i] + h[i+1]) / 3.0;
        if (i < n_inner - 1) {
            A(i, i+1) = h[i+1] / 6.0;
            A(i+1, i) = h[i+1] / 6.0; // Symmetric
        }
    }

    // Pre-factorize A since we will solve for multiple RHS (one for each knot)
    // Since A is symmetric positive definite (usually), we can use LLT (Cholesky)
    Eigen::LLT<Eigen::MatrixXd> solver;
    solver.compute(A);
    
    if (solver.info() != Eigen::Success) {
         throw std::runtime_error("Decomposition failed for spline system.");
    }

    // 2. We need to evaluate the spline for each basis vector e_j (j=0..n_knots-1)
    // However, instead of solving for M individually, we can solve for all M at once.
    // Let Y be the Identity matrix (n_knots x n_knots).
    // The RHS for the system A * M = B * Y is structured.
    // The equation for inner knot i (which is index i+1 in knots array):
    // (y[i+2] - y[i+1])/h[i+1] - (y[i+1] - y[i])/h[i]
    
    // Let's construct the matrix B that maps Y to the RHS vectors.
    // RHS_i = (Y_{i+2} - Y_{i+1})/h_{i+1} - (Y_{i+1} - Y_i)/h_i
    // RHS is (n_knots-2) x n_knots
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_inner, n_knots);
    for (long i = 0; i < n_inner; ++i) {
        // Corresponds to knot index k = i + 1
        // Terms involving y[i], y[i+1], y[i+2]
        // y[i] coeff:   1/h[i]
        // y[i+1] coeff: -1/h[i] - 1/h[i+1]
        // y[i+2] coeff: 1/h[i+1]
        
        B(i, i)     = 1.0 / h[i];
        B(i, i+1)   = -1.0 / h[i] - 1.0 / h[i+1];
        B(i, i+2)   = 1.0 / h[i+1];
    }
    
    // Solve for M_inner: A * M_inner = B * I = B
    Eigen::MatrixXd M_inner = solver.solve(B); // (n_knots-2) x n_knots
    
    // Full M matrix including natural boundary conditions (0 at ends)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_knots, n_knots);
    M.block(1, 0, n_inner, n_knots) = M_inner;
    
    // 3. Evaluate the splines at x
    Eigen::MatrixXd N(n_x, n_knots);
    
    // For efficiency, sort x or just search. Assuming x is sorted is faster, but we shouldn't assume.
    // Using binary search (std::upper_bound) for each x.
    
    // Y is effectively I, so y_j (the j-th basis) is 1 at knot j, 0 elsewhere.
    
    for (long i = 0; i < n_x; ++i) {
        double val = x[i];
        
        // Find interval
        // We need index k such that knots[k] <= val <= knots[k+1]
        // If val < knots[0] or val > knots[end], we extrapolate linearly using the first/last interval
        
        long k = 0;
        if (val < knots[0]) {
            k = 0; // Use first interval for extrapolation
        } else if (val >= knots[n_knots - 1]) {
            k = n_knots - 2; // Use last interval for extrapolation
        } else {
            // Binary search
            auto it = std::upper_bound(knots.data(), knots.data() + n_knots, val);
            k = std::distance(knots.data(), it) - 1;
            if (k < 0) k = 0;
            if (k >= n_knots - 1) k = n_knots - 2;
        }
        
        double hk = h[k];
        double t = (val - knots[k]) / hk; // Normalized parameter [0, 1] usually, but can be <0 or >1 for extrapolation
        
        // Cubic Hermite / Spline formula
        // S(x) = (1-t)*y_k + t*y_{k+1} + h_k^2/6 * [ ( (1-t)^3 - (1-t) ) * M_k + ( t^3 - t ) * M_{k+1} ]
        
        double term1 = (1.0 - t);
        double term2 = t;
        double term3 = (std::pow(1.0 - t, 3) - (1.0 - t)) * hk * hk / 6.0;
        double term4 = (std::pow(t, 3) - t) * hk * hk / 6.0;

        // The value S(x) is a linear combination of y_j's.
        // We want to fill the i-th row of N with the coefficients of y_j.
        // S(x) = term1 * y_k + term2 * y_{k+1} + term3 * M_k + term4 * M_{k+1}
        // M_k is the k-th row of M. M_k = \sum_j M_{kj} y_j
        
        // N[i, :] = term1 * e_k + term2 * e_{k+1} + term3 * M[k, :] + term4 * M[k+1, :]
        
        N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
        N(i, k)   += term1;
        N(i, k+1) += term2;
    }

    return N;
}

PYBIND11_MODULE(_spline_extension, m) {
    m.doc() = "C++ implementation of SplineFitter core components"; // optional module docstring
    m.def("compute_natural_spline_basis", &compute_natural_spline_basis, "Compute the Natural Cubic Spline Basis matrix");
}
