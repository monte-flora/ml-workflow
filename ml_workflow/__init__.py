from .calibrated_hyperopt_cv import CalibratedHyperOptCV
from .preprocess.preprocess import PreProcessPipeline

import os
__key__ = 'PACKAGE_VERSION'
__version__= os.environ[__key__] if __key__ in os.environ else '1.1.0'
