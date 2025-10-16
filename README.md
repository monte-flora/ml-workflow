# ml_workflow
A scikit-learn-based model that incorporates ML pipelines, hyperparameter optimization, and calibration all within a cross-validation framework. The primary method is the [TunedEstimator](https://github.com/monte-flora/ml-workflow/blob/main/ml_workflow/estimators/tuned_estimator.py). 


## Dependencies 

ml_workflow is compatible with Python 3.8 or newer.  It requires the following packages:

```
numpy 
pandas
scikit-learn
imbalanced-learn
hyperopt
lightgbm
```

After cloning the repo, you can install these packages and the repo itself with the following
```
cd ml_workflow
pip install -e .
```


### Using ml_workflow 
```python
from ml_workflow import TunedEstimator
from sklearn.ensemble import HistGradientBoostingRegressor

def mse_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return mean_squared_error(y, pred)

X,y = make_regression(n_samples=10000, n_features=15, random_state=42)
X = pd.DataFrame(X)

pipeline_kwargs=dict(
    imputer="simple",
    scaler="standard"
    pca=False,
    resample=None,
)

# Note: requires a unique date for each sample
cv = DateBasedCV(n_splits=5, dates=dates, valid_size=0.25)

hyperopt_kwargs = dict(
    search_space = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_depth': [5, 10, None],
        'max_leaf_nodes': [31, 63],
        'l2_regularization': [0.0, 0.1, 1.0],
    },
    optimizer = "random_search", 
    max_evals = 5,
    cv = cv, 
    output_fname = "hyperparam_results.csv"
)

estimator = HistGradientBoostingRegressor(random_state=30) 
tuned_estimator = TunedEstimator(estimator, 
                                 pipeline_kwargs,
                                 hyperopt_kwargs,
                                )
tuned_estimator.fit(X,y)
```
