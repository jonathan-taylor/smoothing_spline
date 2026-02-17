import numpy as np
from .loess import LoessSmoother as LoessSmootherPy
from scatter_smooth import LoessSmoother 

def test_compare_cpp_python_loess():
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.1, n)
    
    span = 0.5
    degree = 1
    
    # Python
    loess_py = LoessSmootherPy(x=x, span=span, degree=degree)
    loess_py.smooth(y)
    
    # C++
    loess_cpp = LoessSmoother(x=x, span=span, degree=degree)
    loess_cpp.smooth(y)
    
    x_new = np.linspace(0, 10, 50)
    
    pred_py = loess_py.predict(x_new)
    pred_cpp = loess_cpp.predict(x_new)
    
    np.testing.assert_allclose(pred_py, pred_cpp, rtol=1e-10, atol=1e-10, err_msg="C++ and Python Loess predictions differ")
    print("Loess C++ vs Python comparison passed!")

    # Test derivatives
    if degree >= 1:
        deriv_py = loess_py.predict(x_new, deriv=1)
        deriv_cpp = loess_cpp.predict(x_new, deriv=1)
        np.testing.assert_allclose(deriv_py, deriv_cpp, rtol=1e-10, atol=1e-10, err_msg="C++ and Python Loess derivatives differ")
        print("Loess C++ vs Python derivatives comparison passed!")

if __name__ == "__main__":
    test_compare_cpp_python_loess()
