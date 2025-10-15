# setup.py - UPDATED VERSION

from setuptools import setup, find_packages
import pathlib
import re

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

def get_version():
    """Get version, with fallback"""
    return '1.1.0'  # Simple and works!

setup(
    name='ml_workflow', 
    version=get_version(),  # Don't import ml_workflow!
    description='A scikit-learn-based model that incorporates ML pipelines, hyperparameter optimization, and calibration all within a cross-validation framework', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/WarnOnForecast/ml_workflow', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',  # Fixed typo
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'scikit-learn>=1.0.2',  # Changed == to >= for flexibility
        'scikit-learn-intelex>=2023.0.1',
        'scikit-image>=0.19',
        'xarray>=0.18',
        'imbalanced-learn',
        'pandas>-1.3.0',
        'hyperopt',
        'lightgbm',
        'numpy>=1.20',  # Add explicit numpy requirement
    ],
    packages=find_packages(),  # Use find_packages() instead of manual list
    python_requires='>=3.8, <4',
    project_urls={
        'Bug Reports': 'https://github.com/monte-flora/ml_workflow/issues',
        'Source': 'https://github.com/monte-flora/ml_workflow',
    },
)