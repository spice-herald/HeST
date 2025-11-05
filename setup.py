from setuptools import setup, find_packages

setup(
    name='HeST',
    version='0.2.0',
    packages=find_packages(include=["HeST", "HeST.*"]),
    install_requires=[ 'numpy', 'scipy', 'detprocess', 'numba'],
    package_data={
        'HeST': ['dispersion_curves/*.txt', 'dispersion_curves/*.csv', 'dispersion_curves/*.npy'],
    },
    include_package_data = True
)
