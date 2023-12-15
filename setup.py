from setuptools import setup, find_packages

setup(
    name='HeST',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[ 'numpy', 'scipy', 'detprocess', 're'],
    package_data={
        'HeST': ['*.npy'],
    }
)
