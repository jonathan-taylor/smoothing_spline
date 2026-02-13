from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# Use pybind11's helpers
# We check for pybind11. If not present, we can't build the extension.
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    has_pybind11 = True
except ImportError:
    has_pybind11 = False

if has_pybind11:
    ext_modules = [
        Pybind11Extension(
            "smoothing_spline._spline_extension",
            ["src/spline_basis.cpp"],
            # Example: passing in the version
            define_macros = [('VERSION_INFO', '"0.0.1"')],
            include_dirs = ["src/eigen"],
        ),
    ]
else:
    # If pybind11 is not installed, we can't build the extension.
    # We might want to just skip it or error out.
    # For this exercise, we assume pybind11 is present as we checked it.
    ext_modules = []

setup(
    name="smoothing_spline",
    # If using pyproject.toml, metadata is handled there usually,
    # but `ext_modules` must be passed here or in `pyproject.toml` configuration
    # (though `pyproject.toml` support for extensions is limited without a backend wrapper).
    # Since we have pyproject.toml using setuptools.build_meta,
    # adding a setup.py allows us to define extensions.
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext} if has_pybind11 else {},
)
