#include "utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace utils {

    double brent_root(std::function<double(double)> func, double a, double b, double tol, int max_iter) {
        double fa = func(a);
        double fb = func(b);
        if (fa * fb > 0) {
             throw std::runtime_error("Root not bracketed: target DF out of feasible range.");
        }
        
        if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
        
        double c = a;
        double fc = fa;
        bool mflag = true;
        double s = 0;
        double d = 0;
        
        for (int i = 0; i < max_iter; ++i) {
            if (std::abs(b - a) < tol) return b;
            if (std::abs(fb) < tol) return b;

            if (fa != fc && fb != fc) {
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                    (c * fa * fb) / ((fc - fa) * (fc - fb));
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            
            double tmp1 = (3 * a + b) / 4;
            bool cond1 = (s < std::min(tmp1, b) || s > std::max(tmp1, b));
            bool cond2 = mflag && (std::abs(s - b) >= (std::abs(b - c) / 2));
            bool cond3 = !mflag && (std::abs(s - b) >= (std::abs(c - d) / 2));
            bool cond4 = mflag && (std::abs(b - c) < tol);
            bool cond5 = !mflag && (std::abs(c - d) < tol);
            
            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2;
                mflag = true;
            } else {
                mflag = false;
            }
            
            double fs = func(s);
            d = c;
            c = b;
            fc = fb;
            
            if (fa * fs < 0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
            
            if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
        }
        return b;
    }

    double brent_min(std::function<double(double)> func, double a, double b, double tol, int max_iter) {
        double x, w, v, fx, fw, fv;
        double d = 0.0, e = 0.0;
        double u, fu;
        const double gold = 0.3819660;
        
        x = w = v = a + gold * (b - a);
        fx = fw = fv = func(x);
        
        for(int iter = 0; iter < max_iter; ++iter) {
            double xm = 0.5 * (a + b);
            double tol1 = tol * std::abs(x) + 1e-10;
            double tol2 = 2.0 * tol1;
            
            if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
                return x;
            }
            
            if (std::abs(e) > tol1) {
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = std::abs(q);
                double etemp = e;
                e = d;
                
                if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                    d = gold * (e = (x >= xm ? a - x : b - x));
                } else {
                    d = p / q;
                    u = x + d;
                    if (u - a < tol2 || b - u < tol2) {
                        d = (xm - x >= 0 ? 1 : -1) * tol1;
                    }
                }
            } else {
                d = gold * (e = (x >= xm ? a - x : b - x));
            }
            
            u = (std::abs(d) >= tol1 ? x + d : x + (d > 0 ? 1 : -1) * tol1);
            fu = func(u);
            
            if (fu <= fx) {
                if (u >= x) a = x; else b = x;
                v = w; w = x; x = u;
                fv = fw; fw = fx; fx = fu;
            } else {
                if (u < x) a = u; else b = u;
                if (fu <= fw || w == x) {
                    v = w; w = u;
                    fv = fw; fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u;
                    fv = fu;
                }
            }
        }
        return x;
    }

}
