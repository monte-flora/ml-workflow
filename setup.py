# Always prefer setuptools over distutils

import setuptools  # this is the "magic" import

import ml_workflow

from numpy.distutils.core import setup, Extension

#from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='ml_workflow', 
    version=ml_workflow.__version__,
    description='A scikit-learn-based model that incorporates ML pipelines, hyperparameter optimization, and calibration all within a cross-validation framework', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/WarnOnForecast/ml_workflow', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists',
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'sklearn',
        'scikit-image>=0.18.1',
        'xarray',
        'imblearn',
        'hyperopt',
        'lightgbm'
    ],
    packages=['ml_workflow', 'ml_workflow.io', 'ml_workflow.preprocess'],  # Required
    python_requires='>=3.8, <4',
    package_dir={'ml_workflow': 'ml_workflow'},
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/WarnOnForecast/ml_workflow/issues',
        'Source': 'https://github.com/WarnOnForecast/ml_workflow',
    },
)