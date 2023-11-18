from setuptools import setup, find_packages

setup(
    name='HeST',
    version='0.1.3',
    packages=find_packages(),
    install_requires=[ 'numpy', 'scipy', 'detprocess'],
    package_data={
        'HeST': ['*.npy'],
    }
)
