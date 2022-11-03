from setuptools import setup, find_packages

setup(
    name='differentiable_t1d',
    version='0.1',
    description='A differentiable simulator for Type 1 Diabetes',
    author="Ke Alexander Wang",
    author_email="kaw293@cornell.edu",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["pandas", "numpy", "scipy", "torch"],
)