"""
Hyperparameter Optimization with HyperOpt in scikit-learn format.

This module provides a scikit-learn-compatible wrapper around the hyperopt library
for Bayesian hyperparameter optimization, with fallback support for sklearn's grid and random search.
"""

from hyperopt.early_stop import no_progress_loss
from hyperopt import fmin, tpe, atpe, hp, SparkTrials, STATUS_OK, Trials,space_eval

from joblib import delayed, Parallel
import numpy as np 
import pandas as pd

#sklearn 
from sklearn.model_selection._validation import check_cv
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.base import is_classifier
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)

from timeit import default_timer as timer
import ast


def _fit(estimator, X, y, sample_weight): 
    return estimator.fit(X, y, sample_weight)

def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)

class HyperOptCV:
    """
    Scikit-learn style hyperparameter optimization using Bayesian methods (HyperOpt).
    
    This class wraps the hyperopt library to provide Tree-structured Parzen Estimator (TPE)
    and Adaptive TPE optimization with a familiar sklearn interface. It also supports 
    traditional grid and random search as fallback options.
    
    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator (or pipeline) that implements fit() and score() methods.
        This is the model whose hyperparameters will be optimized.
        
    search_space : dict
        Hyperparameter search space. Keys are parameter names (str), values define the 
        search range:
        
        For TPE/ATPE optimizers:
            - Lists: Converted to hp.choice() for categorical selection
            - HyperOpt distributions: Use hp.uniform(), hp.loguniform(), etc. directly
            
        For grid/random search:
            - Lists or arrays: Treated as discrete values to try
            
        Example for TPE:
            {'max_depth': hp.quniform('max_depth', 2, 20, 1),
             'learning_rate': hp.loguniform('learning_rate', -5, 0),
             'n_estimators': [100, 200, 500]}
             
        Example for grid search:
            {'max_depth': [5, 10, 15],
             'learning_rate': [0.01, 0.1, 1.0]}
        
    optimizer : {'tpe', 'atpe', 'grid_search', 'random_search'}, default='tpe'
        Optimization strategy:
        - 'tpe': Tree-structured Parzen Estimator (Bayesian optimization)
        - 'atpe': Adaptive TPE (adjusts to optimization progress)
        - 'grid_search': Exhaustive grid search (uses sklearn.GridSearchCV)
        - 'random_search': Random sampling (uses sklearn.RandomizedSearchCV)
        
    max_evals : int, default=100
        Maximum number of hyperparameter configurations to evaluate.
        For Bayesian methods, this is the optimization budget.
        For random_search, this is the number of random samples.
        For grid_search, this is ignored (all combinations are tried).
        
    patience : int, default=10
        Early stopping patience for Bayesian optimization. Stops if no improvement
        in validation loss for this many consecutive iterations. Only applies to
        TPE/ATPE optimizers.
        
    scorer : callable or None, default=None
        Scoring function of signature scorer(estimator, X, y). If None, uses the
        estimator's default score() method. 
        
        Example:
            from sklearn.metrics import make_scorer, roc_auc_score
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        
    n_jobs : int, default=1
        Number of parallel jobs for cross-validation. -1 uses all processors.
        
    cv : int, cross-validation generator, or iterable, default=None
        Cross-validation strategy:
        - None: Uses 5-fold stratified CV for classifiers, 5-fold CV for regressors
        - int: Number of folds in (Stratified)KFold
        - CV splitter object: e.g., KFold(n_splits=5, shuffle=True)
        - Iterable: Yields (train_idx, test_idx) tuples
        
    output_fname : str or None, default=None
        (Currently unused) Path to save optimization results.
    
    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found during optimization.
        
    dataframe_ : list of dict
        History of all evaluated configurations with their scores and metadata.
        
    optimizer_ : str
        The optimizer method being used.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from hyperopt import hp
    >>> 
    >>> # Create sample data
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>> 
    >>> # Define model and search space
    >>> rf = RandomForestClassifier(random_state=42)
    >>> search_space = {
    ...     'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
    ...     'max_depth': hp.quniform('max_depth', 3, 15, 1),
    ...     'min_samples_split': [2, 5, 10]
    ... }
    >>> 
    >>> # Optimize hyperparameters
    >>> optimizer = HyperOptCV(
    ...     estimator=rf,
    ...     search_space=search_space,
    ...     optimizer='tpe',
    ...     max_evals=50,
    ...     patience=10,
    ...     cv=5,
    ...     n_jobs=-1
    ... )
    >>> optimizer.fit(X, y)
    >>> print(f"Best parameters: {optimizer.best_params_}")
    
    Notes
    -----
    - For imbalanced datasets, consider using sample_weight parameter in fit()
    - TPE/ATPE are generally more efficient than grid/random search for large spaces
    - The loss being minimized is the negative of the scorer (or negative score())
    - Results are stored in dataframe_ for post-hoc analysis
    
    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive grid search
    sklearn.model_selection.RandomizedSearchCV : Random parameter search
    hyperopt.fmin : Core HyperOpt optimization function
    
    References
    ----------
    .. [1] Bergstra, J., Yamins, D., Cox, D. (2013). "Making a Science of Model Search:
           Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures."
           ICML 2013.
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
        self._iteration = 0
        self._n_jobs = n_jobs
        self._cv = cv
        self.dataframe_ = []
    
    
    def _convert_search_space(self, search_space):
        """
        Convert search space to optimizer-specific format.
        
        For TPE/ATPE: Converts lists to hp.choice() for categorical variables.
        For grid/random search: Prefixes keys with 'model__' for pipeline compatibility.
        
        Parameters
        ----------
        search_space : dict
            Original search space definition.
            
        Returns
        -------
        dict
            Converted search space in optimizer-specific format.
        """
        for p, values in search_space.items():
            if self.optimizer_ in ['tpe', 'atpe']:
                if isinstance(values, list): 
                    return {p: hp.choice(p, values) for p,values in search_space.items()}
                else: 
                    return search_space
            else:
                return {f'model__{p}': values for p,values in search_space.items()}

    
    def fit(self, X, y, groups=None, sample_weight=None):
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
            
        sample_weight: array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples. 
            If not provided, then each sample is given unit weight.
            
        Returns
        -------
        self : object
            Returns self with best_params_ attribute set.
            
        Notes
        -----
        After fitting, access optimization results via:
        - self.best_params_: Best hyperparameters found
        - self.dataframe_: Full optimization history
        - self._trials: Raw HyperOpt trials object (for TPE/ATPE only)
        """    
        self.sample_weight = sample_weight
        
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
            this_estimator = clone(self._estimator)
            if self.optimizer_ == 'grid_search':
                # Perform a grid search over the hyperparameters using cross-validation
                clf = GridSearchCV(this_estimator, 
                               self._search_space,
                               scoring=self._scorer,
                               n_jobs=self._n_jobs,
                               refit=False,
                               cv=self._cv,
                              )
            elif self.optimizer_ == 'random_search':
                # Perform a random search over the hyperparameters using cross-validation
                clf = RandomizedSearchCV(this_estimator, 
                               self._search_space,
                               n_iter = self._max_evals,
                               scoring=self._scorer,
                               n_jobs=self._n_jobs,
                               refit=False,
                               cv=self._cv,
                                        
                              )
            
            fit_params = {'model__sample_weight' : self.sample_weight}
            clf.fit(self._X, self._y, groups=self._groups, **fit_params)
            
            # Convert sklearn results to consistent format
            self._convert_sklearn_results(clf)

            # Remove the 'model' part of the keys:
            self.best_params_ = {param.split('__')[1] : val for param, val in clf.best_params_.items()}
   
    def _convert_sklearn_results(self, search):
        """
        Convert sklearn CV results to HyperOpt-consistent format.
        
        Parameters
        ----------
        search : GridSearchCV or RandomizedSearchCV
            Fitted sklearn search object.
        """
        df = pd.DataFrame(search.cv_results_)
        
        # Select relevant columns
        cols_to_keep = ['mean_fit_time', 'mean_test_score', 'std_test_score', 'rank_test_score']
        cols_to_keep += [col for col in df.columns if 'param_' in col]
        df = df[cols_to_keep]
        
        # Rename to match HyperOpt format
        df.rename(columns={
            'mean_fit_time': 'train_time', 
            'mean_test_score': 'loss',
            'std_test_score': 'loss_variance',
        }, inplace=True)
        
        self.dataframe_ = df


    def write_to_frame(self, data, params):
        """
        Convert the output from objective to a new row in the dataframe. 
        """
        for param in params.keys():
            if isinstance(params[param], dict):
                p = params[param]
                dict_list = [':'.join((str(k), str(i))) for k,i in p.items()]
                new_p = ','.join(dict_list)
                data[param] = new_p
            else:
                data[param] = params[param]

        self.dataframe_.append(data)
        
    def objective(self, params):
        """
        Objective function for HyperOpt optimization.
        
        Evaluates a hyperparameter configuration using cross-validation and
        returns the loss (negative score) to be minimized.
        
        Parameters
        ----------
        params : dict
            Hyperparameter configuration to evaluate.
            
        Returns
        -------
        dict
            Results dictionary with keys:
            - 'loss': Mean CV score (to minimize)
            - 'loss_variance': Variance of CV scores
            - 'iteration': Current iteration number
            - 'params': Parameters used
            - 'train_time': Time taken for evaluation
            - 'status': STATUS_OK for successful evaluation
        """
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
                                  params= {'model__sample_weight' : self.sample_weight}, error_score='raise'
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
    
    