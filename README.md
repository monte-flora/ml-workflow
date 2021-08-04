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

dates = train_df['date']

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
                                    cross_val_kwargs = {'n_splits' : 5, 'dates' : dates, 'valid_size' : 30},
                                  )
```
