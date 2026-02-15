#pragma once
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <vector>

namespace py = pybind11;

namespace bspline {
    void eval_bspline_basis(double x, int k, const Eigen::VectorXd& knots, int& span_index, Eigen::VectorXd& N, int deriv=0);
}

class BSplineFitter {
    int order_; Eigen::VectorXd knots_, coeffs_, weights_, x_; int n_basis_; Eigen::MatrixXd AB_template_, Omega_band_;
public:
    BSplineFitter(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& i_k, py::object w_obj, int order = 4);
    void compute_NTWN();
    void compute_penalty_matrix();
    Eigen::VectorXd eval_basis(double x_val, int deriv=0);
    Eigen::VectorXd get_knots();
    Eigen::MatrixXd get_NTWN();
    Eigen::MatrixXd get_Omega();
    void fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval);
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_n, int deriv=0);
};
