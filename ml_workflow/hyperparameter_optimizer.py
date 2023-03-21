# A python class for the hyperopt package in a scikit-learn-friendly format. 

from hyperopt.early_stop import no_progress_loss
from hyperopt import fmin, tpe, atpe, hp, SparkTrials, STATUS_OK, Trials,space_eval

from joblib import delayed, Parallel
import numpy as np 
import pandas as pd

#sklearn 
from sklearn.model_selection._validation import check_cv
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, GridSearchCV
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.base import is_classifier
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)

from timeit import default_timer as timer
import ast


def _fit(estimator, X, y): 
    return estimator.fit(X, y)

def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)

class HyperOptCV:
    """
    estimator : estimator instance, default=None
        The estimator to perform hyperparameter optimization on. 
        
    search_space : dict
        Keys are the parameter names (strings) and values are lists or arrays of 
        some combination of integers, floats, or categorical (strings/None) that are 
        valid values for the given parameter. Represents search space
        over parameters of the provided estimator.
        
    optimizer : 'tpe' or 'atpe', 'grid'
        The optimizer option for hyperopt. If 'grid', then the uses the GridSearchCV in 
        sklearn rather than bayesian. 
        
    max_evals : integer 
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
        
    patience : integer 
        The number of iterations in which the cross-validation score is expected to
        improve, otherwise there is early stopping. 
        
    scorer : callable of form (estimator, X, y) (default is None)
        
    n_jobs : integer 
        Parallelization for the cross-validation 
        
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. 
        
    
    """
    def __init__(self, estimator, search_space, optimizer='tpe', max_evals=100, patience=10, 
                 scorer=None, n_jobs=1, cv=None, output_fname=None):
        
        self._estimator = estimator
        self.optimizer_ = optimizer
        self._search_space = self._convert_search_space(search_space)

        self._algo = tpe.suggest if optimizer == 'tpe' else atpe.suggest
        self._max_evals = max_evals
        self._patience = patience
        self._trials = Trials()
        self._scorer = scorer 
        self._iteration=0
        self._n_jobs = n_jobs
        self._cv =  cv
    
        self.dataframe_ = []
    
    
    def _convert_search_space(self, search_space):
        """
        Converts a parameter grid to a hyperopt-friendly format if provide a list.
        """
        for p, values in search_space.items():
            if self.optimizer_ in ['tpe', 'atpe']:
                if isinstance(values, list): 
                    return {p: hp.choice(p, values) for p,values in search_space.items()}
                else: 
                    return search_space
            else:
                return {f'model__{p}': values for p,values in search_space.items()}

    
    def fit(self, X, y, groups=None):
        """Find the best hyperparameters using the hyperopt package
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        
        """
        # Using early stopping in the error minimization. Need to have 1% drop in loss every 8-10 count (varies)
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, allow_nd=True)
        X, y = indexable(X, y)
        
        self._cv =  check_cv(self._cv, y, classifier=is_classifier(self._estimator))
        
        self._X = X 
        self._y = y 
        self._groups = groups 
        
        if self.optimizer_ in ['tpe', 'atpe']:
        
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
            
        else:
            # Perform a grid search over the hyperparameters using cross-validation
            this_estimator = clone(self._estimator)
            clf = GridSearchCV(this_estimator, 
                               self._search_space,
                               scoring=self._scorer,
                               n_jobs=self._n_jobs,
                               refit=False,
                               cv=self._cv,
                              )
            clf.fit(self._X, self._y, groups=self._groups)
            
            # Renaming these results to be consistent with the output from the 
            # hyperopt results. 
            self.dataframe_ = pd.DataFrame(clf.cv_results_)
            cols_to_keep = ['mean_fit_time', 'mean_test_score', 'std_test_score', 'rank_test_score']

            cols_to_keep += [col for col in self.dataframe_.columns if 'param_' in col]
            self.dataframe_ = self.dataframe_[cols_to_keep]

            rename_map = {'mean_fit_time' : 'train_time', 
              'mean_test_score' : 'loss',
              'std_test_score' : 'loss_variance', 
              
             }

            self.dataframe_.rename(rename_map, axis=1, inplace=True)

            # Remove the 'model' part of the keys:
            self.best_params_ = {param.split('__')[1] : val for param, val in clf.best_params_.items()}
            

    def write_to_frame(self, data, params):
        """
        Convert the output from objective to a new row in the dataframe. 
        """
        for param in params.keys():
            data[param] = params[param]
            
        self.dataframe_.append(data)
        
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

        scores = cross_val_score(this_estimator, self._X, self._y, groups=self._groups, 
                                  scoring=self._scorer, cv=self._cv, n_jobs=self._n_jobs, 
                                  )
            
        run_time = timer() - start
        # Loss must be minimized (using NAUPDC as the metric!)
        loss = np.nanmean(scores)
        loss_variance = np.nanvar(scores, ddof=1)
        
        # Dictionary with information for evaluation
        data = {'loss': loss, 
                'loss_variance': loss_variance, 
                'iteration': self._iteration, 
                'train_time': run_time}
        
        self.write_to_frame(data, params)

        return {'loss': loss, 'loss_variance': loss_variance, 'iteration': self._iteration, 'params' : params,
                'train_time': run_time, 'status': STATUS_OK}



def convergence_plot(fname, param):
    df = pd.read_feather(fname)
    df.sort_values(by=param)
    df.plot(param, 'loss')
    
    