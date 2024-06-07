try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import re
from pathlib import Path

def version(root_path):
    """Returns the version taken from __init__.py

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package

    Reference
    ---------
    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    version_path = root_path.joinpath('plpm', '__init__.py')
    with version_path.open() as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

root_path = Path(__file__).parent
VERSION = version(root_path)

config = {
    'name': 'plpm',
    'packages': find_packages(),
    'description': 'Modeling the Cardiovascular Circulation Using PINN',
    'author': [
        {'name': "Ehsan Naghavi", 'email': "naghavis@msu.edu"},
        ],
    'version': VERSION,
    'install_requires': ['numpy', 'torch', 'matplotlib', 'SALib', 'scipy', 'jupyter', 'notebook'],
    'license': 'MIT license',
    'scripts': [],
    'include_package_data': True,
    'package_data': {'': ['data']},
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
