import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import platform
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)
ext_modules = [
    Extension(name='poly_cpu',
              sources=['poly_cpu_func.cpp', 'poly_cpu.pyx'],
              language='c++',
              include_dirs = [numpy_include],
              )
]
setup(
      ext_modules=cythonize(ext_modules, language_level=3),
      cmdclass={'build_ext': custom_build_ext},
)