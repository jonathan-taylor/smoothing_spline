from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "smoothing_spline._spline_extension",
        [
            "src/main.cpp",
            "src/utils.cpp",
            "src/natural_spline.cpp",
            "src/reinsch.cpp",
            "src/bspline.cpp"
        ],
        # Example: passing in the version
        define_macros = [('VERSION_INFO', '"0.0.1"')],
        include_dirs = ["src/eigen"],
    ),
]

setup(
    name="smoothing_spline",
    # If using pyproject.toml, metadata is handled there usually,
    # but `ext_modules` must be passed here or in `pyproject.toml` configuration
    # (though `pyproject.toml` support for extensions is limited without a backend wrapper).
    # Since we have pyproject.toml using setuptools.build_meta,
    # adding a setup.py allows us to define extensions.
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
