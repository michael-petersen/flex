from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="flex.laguerre_cython",
        sources=["src/flex/laguerre_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="flex",
    packages=["flex"],
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        language_level="3",
        annotate=True,  # generates .html for optimisation insight
    ),
)
