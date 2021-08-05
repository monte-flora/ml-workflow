##############################################
# TrainModels 
##############################################
import csv
import pandas as pd
import numpy as np
from os.path import join, exists
import os
from imblearn.pipeline import Pipeline
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)
from sklearn.model_selection import KFold
import joblib
from timeit import default_timer as timer
import ast

import sys
sys.path.append('/Users/monte.flora/Desktop/PHD_PLOTS')

from .common.calibration import CalibratedClassifierCV
from .preprocess.preprocess import PreProcessPipeline
from .io.cross_validation_generator import DateBasedCV
from .ml_methods import norm_aupdc, norm_csi
from .my_hyperopt.hyperopt.early_stop import no_progress_loss
from .my_hyperopt.hyperopt import fmin, tpe, atpe, hp, SparkTrials, STATUS_OK, Trials,space_eval
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

os.environ['KMP_WARNINGS'] = '0'

def norm_aupdc_scorer(model, X, y, **kwargs):
    known_skew = kwargs.get('known_skew', np.mean(y))

    predictions = model.predict_proba(X)[:,1]
    score = norm_aupdc(y, predictions, known_skew=known_skew)
    return 1.0 - score

def norm_csi_scorer(model, X, y, **kwargs):
    known_skew = kwargs.get('known_skew', np.mean(y))

    predictions = model.predict_proba(X)[:,1]
    score = norm_csi(y, predictions, known_skew=known_skew)
    return 1.0 - score


class CalibratedPipelineHyperOptCV(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    """
    This class takes X,y as inputs and returns 
    a ML pipeline with optimized hyperparameters (through k-folds cross-validation)  
    that is also calibrated using isotonic regression. 
    
    Parameters:
    ---------------------------
        estimator : unfitted callable classifier (likely from scikit-learn) 
                    that implements ``fit`` and ``predict_proba`` methods
        
        param_grid : dict of hyperparameters and the grid to search over. 
        
        E.g., param for sklearn.ensemble.RandomForestClassiferm could be 
        
            param_grid = {  'n_estimators' : [100,150,300,400,500], 
                'max_depth' : [6,8,10,15,20],
                'max_features' : [5,6,8,10],
                'min_samples_split' : [4,5,8,10,15,20,25,50],
                'min_samples_leaf' : [4,5,8,10,15,20,25,50],
             }
        For simplicity, this method only implements choices and not ranges. So the
        ranges must be defined as lists. 
        
        
        scorer : callable object/function that takes inputs (model,X,y, **scorer_kwargs).
                 Used as the scorer for the hyperopt package to optimize hyperparameters, 
                 so the scorer must return a loss metric (where a lower value is better).
                 If the score is rank-based (like AUC), then have scorer
                 return 1 - AUC. 
                  
        scorer_kwargs : Any keyword arguments to pass into scorer. 
        
        imputer : ``simple``,  ``iterative``, or None. If not None, then an imputer is included 
                   in the pipeline. Imputers are used to replace missing values. 
                   
                   - ``simple`` uses sklearn.impute.SimpleImputer
                   - ``iterative`` uses sklearn.impute.IterativeImputer
                   
        scaler : ``standard``,  ``robust``, or None. 
                If not None, then 
                the dataset is scaled (useful for regression models when the 
                scale of the predictors varies)
                
                - ``standard`` uses sklearn.preprocessing.StandardScaler
                - ``robust`` uses sklearn.preprocessing.RobustScaler
        
        resample : ``under``, ``over``, or None. 
                    If not None, then the dataset
                   is resampled to balance the dataset. 
                   
                - ``under`` uses imblearn.under_sampling.RandomUnderSampler
                - ``over`` uses imblearn.over_sampling.RandomOverSampler
                   
        local_dir : str/path or None 
                Filename to save the results of the hyperparameter optimization.  
                Directory where the results of the hyperparameter optimization are stored.
                Typically, ~/hyperopt_results.pkl
        
        max_iter : int 
            Number of iterations for early stopping in the hyperopt. The system will check whether the 
            performance has improved within max_iter, otherwise the process stops. 
        
        n_jobs : int (default=1)
            Number of processors for processing the cross-validation in parallel
            
        hyperopt : ``atpe`` or ``tpe``
                Options for the hyperopt python package. This is the
                optimization algorithm used. 

                - ``tpe`` Tree of Parzen Estimators (TPE)
                - ``atpe`` Adaptive TPE
        
        cross_val : ``date_based`` or ``kfold``. 
                
        cross_val_kwargs : dict 
            - n_splits : int
                Number of cross-validation folds 
            
            - dates,  shape = (n_samples,)
                Used for date-based cross-validation generation 
                that recognize the autocorrelation within meteorological data. This column could
                days, month, etc. shape = (n_samples,)
            
            - valid_size : int or float between 0 and 1
                If int, then interpreted as the number of dates to include 
                in each validation dataset. E,g, if valid_size=5 and 
                dates is months, then each validation dataset will 
                include 5 months 
            
            
    Attributes:
    --------------------
        estimator_ : a fit classifier model 

    """
    def __init__(self, estimator, param_grid, imputer='simple', scaler='standard', 
                resample = None, local_dir=os.getcwd(), n_jobs=1, max_iter=10, 
                scorer=norm_csi_scorer, cross_val='kfolds', cross_val_kwargs={'n_splits': 5}, 
                 hyperopt='atpe', scorer_kwargs={},):
        
        self.imputer = imputer
        self.scaler = scaler
        self.resample = resample
        # Build components of the pipeline 
        steps = self.bulid_pipeline()
        steps.append(('model', estimator))
        
         # INITIALIZE THE PIPELINE 
        self.pipe = Pipeline(steps)
        self.algo=atpe.suggest if hyperopt == 'atpe' else tpe.suggest
        
        self.cross_val = cross_val 
        self.cross_val_kwargs = cross_val_kwargs
        self.scorer_kwargs = scorer_kwargs
        self.n_jobs = n_jobs
        self.param_grid = self._convert_param_grid(param_grid)
        self.scorer = scorer
        self.MAX_EVALS = 100
        self.max_iter = max_iter
        self.trials = Trials()
        self.ITERATION = 0

        self.hyperparam_result_fname = join(local_dir, 'hyperopt_results.pkl')
        if self.hyperparam_result_fname is not None:
            # File to save first results
            of_connection = open(self.hyperparam_result_fname, 'w')
            self.writer = csv.writer(of_connection)
    
            # Write the headers to the file
            self.writer.writerow(['loss', 'params', 'iteration', 'train_time'])
            of_connection.close()
    
    
    def fit(self, X, y):
        """
        Fit the estimator 
        
        Parameters:
        ----------------
            X, pd.DataFrame, shape : (n_samples, n_features)
            y, shape: (n_samples, )
        
        """
        self.known_skew = np.mean(y)
        self.X = X
        self.y = y 
        self.init_cv()
    
        self.estimator_ = CalibratedClassifierCV(self.pipe, 
                                                 cv=self.cv, 
                                                 method='isotonic', 
                                                 n_jobs=self.n_jobs)

        self._find_best_params()
        
        # Fit the model with the optimal hyperparamters
        self.final_estimator_ = clone(self.estimator_)
        self.final_estimator_.set_params(**self.best_params)
        self.final_estimator_.fit(X, y)
        
        
        if self.hyperparam_result_fname is not None:
            self.convert_tuning_results(self.hyperparam_result_fname)
        
        self.writer = 0
        
    def predict_proba(self,X):
        return self.final_estimator_.predict_proba(X)
    
   
    def save(self, fname):
        model_dict = {
                    'model' : self,
                    'features': list(self.X.columns),
                    'n_features':len(list(self.X.columns)),
                    'resample' : self.resample,
                    'scaler' : self.scaler,
                    'n_training_examples' : len(self.X),
                    'hyperparameters' : self.best_params, 
                    'skew' : self.known_skew, 
                    }
        
        joblib.dump(model_dict, fname)

    def _convert_param_grid(self, param_grid):
        return {f'base_estimator__model__{p}': hp.choice(p, values) for p,values in param_grid.items()}

    def _find_best_params(self, ):
        """Find the best hyperparameters using the hyperopt package"""
        # Using early stopping in the error minimization. Need to have 1% drop in loss every 8-10 count (varies)
        best = fmin(self.objective,
            self.param_grid,
            algo=self.algo,
            max_evals=self.MAX_EVALS,
            trials=self.trials,
            early_stop_fn=no_progress_loss(iteration_stop_count=self.max_iter,
                percent_increase=1.0),
            )

        # Get the values of the optimal parameters
        best_params = space_eval(self.param_grid, best)

        self.best_params = best_params
        
        
    def init_cv(self,):
        """Initialize the cross-validation generator"""
        # INITIALIZE MY CUSTOM CV SPLIT GENERATOR
        n_splits = self.cross_val_kwargs.get('n_splits')
        if self.cross_val == 'date_based':
            dates = self.cross_val_kwargs.get('dates', None)
            valid_size = self.cross_val_kwargs.get('valid_size', None)
            if dates is None:
                raise KeyError('When using cross_val = "date_based", must provide a date column in cross_val_kwargs')
            else:
                self.cv = DateBasedCV(n_splits=n_splits, dates=dates, y=self.y, valid_size=valid_size)
        else:
            self.cv = KFold(n_splits=n_splits)
            
        
    def bulid_pipeline(self,):
        # BULID THE TRANSFORMATION PORTION OF THE ML PIPELINE 
        preprocess = PreProcessPipeline(
                        imputer = self.imputer,
                        scaler = self.scaler,
                        resample = self.resample,
        )

        steps = preprocess.get_steps()
        return steps
    
    def convert_tuning_results(self, fname):
        """ Convert the hyperopt results to pickle file """
        df = pd.read_csv(fname)
        params = ast.literal_eval(df['params'].iloc[0]).keys()
        params = [p.replace('base_estimator__model__','') for p in params]
        data =  [ast.literal_eval(df['params'].iloc[i]) for i in range(len(df))]
        results = []
        for p in params:
            df[p] = [data[i][f'base_estimator__model__{p}'] for i in range(len(data))]
        df = df.drop(columns='params')

        df.to_pickle(fname)
        return df

    def objective(self, params):
        """Objective function for Hyperparameter Optimization"""
        # Keep track of evals
        self.ITERATION += 1
        start = timer()
        
        this_estimator = clone(self.estimator_)       
        this_estimator.set_params(**params)

        # Perform n_folds cross validation
        this_estimator.fit(self.X, self.y)

        # The fit estimators were fit on the training folds within clf
        scores = [self.scorer(model,self.X.iloc[test,:], self.y[test], **self.scorer_kwargs) 
                          for model, (_, test) in zip(this_estimator.fit_estimators_, self.cv.split(self.X, self.y))]

        run_time = timer() - start
        # Loss must be minimized (using NAUPDC as the metric!)
        loss = np.nanmean(scores)
        
        # Dictionary with information for evaluation
        if self.hyperparam_result_fname is not None:
            # Write to the csv file ('a' means append)
            of_connection = open(self.hyperparam_result_fname, 'a')
            self.writer = csv.writer(of_connection)
            self.writer.writerow([loss, params, self.ITERATION, run_time])

        return {'loss': loss, 'iteration': self.ITERATION, 'params' : params,
                'train_time': run_time, 'status': STATUS_OK}

    


