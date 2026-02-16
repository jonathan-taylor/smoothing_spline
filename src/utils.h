#pragma once
#include <functional>

namespace utils {
    double brent_root(std::function<double(double)> func, double a, double b, double tol=1e-6, int max_iter=100);
    double brent_min(std::function<double(double)> func, double a, double b, double tol=1e-5, int max_iter=100);
}

extern "C" {
}
