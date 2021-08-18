# ml_workflow
A scikit-learn-based model that incorporates ML pipelines, hyperparameter optimization, and calibration all within a cross-validation framework. Documents for the code is provided in calibrated_pipeline_hyperopt_cv.py. 



## Dependencies 

ml_workflow is compatible with Python 3.6 or newer.  It requires the following packages:

```
numpy 
pandas
scikit-learn
imblearn
hyperopt
lightgbm
```

### Using ml_workflow 
```python
import sys, os 
current_dir = os.getcwd()
path = os.path.dirname(current_dir)
sys.path.append(path)

from ml_workflow.calibrated_pipeline_hyperopt_cv import CalibratedPipelineHyperOptCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd 
```

This is an example for training a random forest for predicting sub-freezing road surface temperatures. 
Our goal is to optimize a random forest by finding the optimal hyperparameters and then calibrating it. 
The following code with create an ML pipeline internally to pre-process the data (imputations, scaling, 
resampling, pca transformation). Next, it uses a Bayesian approach to determine hyperparameters in a 
cross-validation framework. CalibratedPipelineHyperOptCV is capable of a date-based cross-validation 
where we would want to account for autocorrelations between different dates. E.g., Data from one date 
should not also be in both the training and validation dates. In this example, I'm using 5-fold cross-validation
where I want at least 30 dates worth of data per validation fold. 

The optimal hyperparameters uses a loss metric, which is defined by the end-user. In this example, 
I'm using the normalized critical success index. 

```python
X,y = make_classification(n_samples=100000, random_state=42, class_sep=0.7)
X = pd.DataFrame(X)
```

```python
# Create a hyperparameter grid to search over. In this case, 
# I am searching over hyperparameters from a random forest. 
param_grid = {  'n_estimators' : [100,150,300,400,500], 
                'max_depth' : [6,8,10,15,20],
                'max_features' : [5,6,8,10],
                'min_samples_split' : [4,5,8,10,15,20,25,50],
                'min_samples_leaf' : [4,5,8,10,15,20,25,50],
             }

# Initialize the estimator that will be using.
estimator = RandomForestClassifier(n_jobs=12, random_state=30, criterion = 'entropy',) 

clf = CalibratedPipelineHyperOptCV( base_estimator = estimator,  
                                    param_grid = param_grid,
                                    imputer=None, 
                                    scaler = None,
                                    resample='under',
                                    n_jobs=1,
                                    max_iter=10,
                                    hyperopt='atpe', 
                                    scorer_kwargs = {'known_skew': np.mean(y)}, 
                                  )
clf.fit(X,y)
```
The hyperparameter optimization results are saved in "hyperopt_results.pkl"

```python
df = pd.read_pickle('hyperopt_results.pkl')
```


