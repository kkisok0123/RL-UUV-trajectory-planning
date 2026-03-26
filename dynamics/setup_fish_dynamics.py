import site
import sys

from setuptools import Extension, setup

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import pybind11


setup(
    name="fish_dynamics",
    version="0.1.0",
    ext_modules=[
        Extension(
            name="fish_dynamics",
            sources=["bindings.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
        )
    ],
)
