from setuptools import Extension, setup


setup(
    name="dynamics_step",
    version="0.1.0",
    ext_modules=[
        Extension(
            name="dynamics_step",
            sources=["dynamics_step.cpp"],
            language="c++",
        )
    ],
)
