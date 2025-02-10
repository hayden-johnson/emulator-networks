from setuptools import find_packages, setup

setup(
    name="emnets",
    version="0.1",
    package_dir={"": "src"},  # Tells setuptools where to find your packages
    packages=find_packages(where="src"),  # Automatically finds packages in `src`
    install_requires=[],  # Add dependencies if necessary
)