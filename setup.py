from setuptools import setup, find_packages

setup(
    name='HeST',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[ 'numpy', 'scipy'],
    package_data={
        'HeST': ['*.npy'],
    }
)
