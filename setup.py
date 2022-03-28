import setuptools

setuptools.setup(
    name        = 'TimeFED',
    version     = '1.1.0',
    description = 'Time-series Forecasting, Evaluation, and Deployment',
    packages    = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'h5py>=2.10.0',
        'mat4py>=0.4.3',
        'mat73>=0.46',
        'matplotlib>=3.3.2',
        'numpy<=1.20',
        'pandas>=1.1.3',
        'pvlib>=0.8.1',
        'pyyaml>=5.4.1',
        'scipy>=1.5.4',
        'seaborn>=0.11.2',
        'scikit-learn==0.24.1',
        'tables>=3.6.1',
        'tqdm>=4.50.1',
        'tsfresh>=0.17.0'
    ],
    python_requires = '~=3.8',
)
