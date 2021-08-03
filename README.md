# ml_workflow
A scikit-learn-based model that incorporates ML pipelines, hyperparameter optimization, and calibration all within a cross-validation framework. Documents for the code is provided in calibrated_pipeline_hyperopt_cv.py. 



## Dependencies 

ml_workflow is compatible with Python 3.6 or newer.  It requires the following packages:

```
numpy 
pandas
scikit-learn
imblearn
```

### Using ml_workflow 


```python
# Add the top of the ml_workflow directory to your system path. 
# Eventually, I will create a setup.py. 
import sys
sys.path.append('/path to /ml_workflow')

from ml_workflow import CalibratedPipelineHyperOptCV
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
estimator = RandomForestClassifier(n_jobs=8, random_state=30, criterion = 'entropy', class_weight = 'balanced') 

def norm_csi_scorer(model, X, y, **kwargs):
    known_skew = kwargs.get('known_skew', np.mean(y))

    predictions = model.predict_proba(X)[:,1]
    score = norm_csi(y, predictions, known_skew=known_skew)
    return 1.0 - score


clf = CalibratedPipelineHyperOptCV( estimator = estimator,  
                                    param_grid = param_grid,
                                    imputer=None, 
                                    scaler = None,
                                    resample='under',
                                    n_jobs=3,
                                    max_iter=10,
                                  hyperopt='atpe', 
                                  n_jobs=3,
                                  scorer=scorer, 
                                  scorer_kwargs = {'known_skew': known_skew}, 
                                  cross_val = 'date_based', 
                                  cross_val_kwargs = {'n_splits' : 5, 'cross_val_column' : months},
                                  )
```