# A python class for the hyperopt package in a scikit-learn-friendly format. 

from hyperopt.early_stop import no_progress_loss
from hyperopt import fmin, tpe, atpe, hp, SparkTrials, STATUS_OK, Trials,space_eval
from joblib import delayed, Parallel
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d


from timeit import default_timer as timer
import ast


def _fit(estimator, X, y): 
    return estimator.fit(X, y)

def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)

class HyperOptCV:
    
    def __init__(self, estimator, search_space, optimizer='atpe', max_evals=100, patience=10, scorer=None, 
                n_jobs=1, cv=KFold(n_splits=5)):
        
        self._estimator = estimator
        self._search_space = self._convert_search_space(search_space)
        self._algo = atpe.suggest if optimizer == 'atpe' else tpe.suggest
        self._max_evals = max_evals
        self._patience = patience
        self._trials = Trials()
        self._scorer = scorer 
        self._iteration=0
        self._n_jobs = n_jobs
        self._cv = cv
    
        self.hyperparam_result_fname = None 
    
    
    def _convert_search_space(self, search_space):
        """
        Converts a parameter grid to a hyperopt-friendly format if provide a list.
        """
        for p, values in search_space.items():
            if isinstance(values, list): 
                return {p: hp.choice(p, values) for p,values in search_space.items()}
            else: 
                return search_space
    
    def fit(self, X, y):
        """Find the best hyperparameters using the hyperopt package"""
        # Using early stopping in the error minimization. Need to have 1% drop in loss every 8-10 count (varies)
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, allow_nd=True)
        X, y = indexable(X, y)
        
        self._X = X 
        self._y = y 
        
        best = fmin(self.objective,
                self._search_space,
                algo=self._algo,
                max_evals=self._max_evals,
                trials=self._trials,
                early_stop_fn=no_progress_loss(iteration_stop_count=self._patience,
                    percent_increase=1.0),
                )

        # Get the values of the optimal parameters
        self.best_params_ = space_eval(self._search_space, best)

    def objective(self, params):
        """Objective function for Hyperparameter Optimization"""
        # Keep track of evals
        self._iteration += 1
        start = timer()
        
        this_estimator = clone(self._estimator)
        
        if hasattr(this_estimator, 'named_steps'):
            # If the estimator is a sklearn pipeline. 
            this_estimator.named_steps['model'].set_params(**params)
        else:
            this_estimator.set_params(**params)

        parallel = Parallel(n_jobs=self._n_jobs)
        # Perform cross validation on the base estimator (no calibration!) 
        fit_estimators_ = parallel(delayed(
                _fit)(clone(this_estimator),self._X[train], self._y[train]) for train, _ in self._cv.split(self._X,self._y))
        
        # The fit estimators were fit on the training folds within clf
        scores = [self._scorer(model,self._X[test,:], self._y[test]) 
                          for model, (_, test) in zip(fit_estimators_, self._cv.split(self._X, self._y))]

        run_time = timer() - start
        # Loss must be minimized (using NAUPDC as the metric!)
        loss = np.nanmean(scores)
        loss_variance = np.nanvar(scores, ddof=1)
        
        # Dictionary with information for evaluation
        if self.hyperparam_result_fname is not None:
            # Write to the csv file ('a' means append)
            of_connection = open(self.hyperparam_result_fname, 'a')
            self.writer = csv.writer(of_connection)
            self.writer.writerow([loss, loss_variance, params, self._iteration, run_time])

        return {'loss': loss, 'loss_variance': loss_variance, 'iteration': self._iteration, 'params' : params,
                'train_time': run_time, 'status': STATUS_OK}

