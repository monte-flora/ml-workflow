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
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.model_selection import KFold
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d

from sklearn.preprocessing import label_binarize, LabelBinarizer, LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import check_cv

import joblib
from joblib import delayed, Parallel
from timeit import default_timer as timer
import ast
import itertools

from .io.cross_validation_generator import DateBasedCV
from .ml_methods import norm_aupdc, norm_csi
from hyperopt.early_stop import no_progress_loss
from hyperopt import fmin, tpe, atpe, hp, SparkTrials, STATUS_OK, Trials,space_eval
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

def _fit(estimator, X, y): 
    return estimator.fit(X, y)

def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)

class CalibratedHyperOptCV(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    """
    This class takes X,y as inputs and returns 
    a ML pipeline with optimized hyperparameters (through k-folds cross-validation)  
    that is also calibrated using isotonic regression. 
    
    Parameters
    ---------------------------
        estimator : unfit callable classifier or regressor (likely from scikit-learn) 
                    that implements a ``fit`` method. 
        
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
        
        cv : ``date_based`` or ``kfold``. 
                
        cv_kwargs : dict 
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
    def __init__(self, estimator, param_grid=None, cal_method='isotonic', 
                 local_dir=os.getcwd(), 
                 n_jobs=1, max_iter=15, 
                 scorer=norm_csi_scorer, 
                 cv='kfolds', cv_kwargs={'n_splits': 5}, 
                 hyperopt='atpe', scorer_kwargs={},):
        
        self.cal_method = cal_method
        self.hyperopt = hyperopt
        self.local_dir = local_dir
        self.pipe = estimator 
        self.estimator = estimator
        
        self.algo=atpe.suggest if hyperopt == 'atpe' else tpe.suggest
        
        self.cv = cv 
        self.cv_kwargs = cv_kwargs
        self.scorer_kwargs = scorer_kwargs
        self.n_jobs = n_jobs
        if param_grid is not None:
            self.param_grid = self._convert_param_grid(param_grid)
        else:
            self.param_grid = None
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
            self.writer.writerow(['loss', 'loss_variance', 'params', 'iteration', 'train_time'])
            of_connection.close()
    
    def fit(self, X, y, params=None):
        """
        Fit the estimator 
        
        Parameters:
        ----------------
            X, pd.DataFrame, shape : (n_samples, n_features)
            y, shape: (n_samples, )
        
        """
        self.features = list(X.columns)
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, allow_nd=True)
        X, y = indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_
        self.known_skew = np.mean(y)
        self.X = X
        self.y = y 
        self.init_cv()
        
        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
                np.any([np.sum(y == class_) < n_folds for class_ in
                        self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        self.calibrated_classifiers_ = []
        cv = check_cv(self.cv, y, classifier=True)
        
        # Perform cross validation on the base estimator (no calibration!)
        if params is None:
            self._find_best_params()
        else:
            self.best_params = params 

        # Fit the model with the optimal hyperparamters
        parallel = Parallel(n_jobs=self.n_jobs)
        this_base_estimator = clone(self.pipe) 
        this_base_estimator.named_steps['model'].set_params(**self.best_params)
        
        # Perform cross validation to get training/validation data for calibration.
        fit_estimators_ = parallel(delayed(
                _fit)(clone(this_base_estimator),self.X[train], self.y[train]) for train, _ in self.cv.split(self.X,self.y))
        
        results = parallel(delayed(
                _predict)(estimator , X[test], y[test]) for estimator, (_, test) in zip(fit_estimators_, self.cv.split(X,y)))

        cv_predictions = [item[0] for item in results ]
        cv_targets = [item[1] for item in results ]

        cv_predictions =  list(itertools.chain.from_iterable(cv_predictions))
        cv_targets =  list(itertools.chain.from_iterable(cv_targets))

        # Clone the original pipeline
        this_estimator = clone(self.pipe)
        # Set the base estimator with the best params
        this_estimator.named_steps['model'].set_params(**self.best_params)
        # Re-fit base_estimator/pipeline on the whole dataset
        refit_estimator = this_estimator.fit(X,y)

        self.base_estimator = refit_estimator 
        # Fit the isotonic regression model (for calibration).
        calibrated_classifier = _CalibratedClassifier(
                    refit_estimator, method=self.cal_method,
                    classes=self.classes_)
        calibrated_classifier.fit(cv_predictions, cv_targets)
        self.calibrated_classifiers_.append(calibrated_classifier)

        try:
            if self.hyperparam_result_fname is not None:
                self.convert_tuning_results(self.hyperparam_result_fname)
        except IndexError:
            print('Issuing with convert the hyperparam output file!') 
    
        # Since we save all attributes, writer needs to be reset 
        self.writer = 0
        
        return self
        
    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
                        force_all_finite=False)

        calibrated_classifier = self.calibrated_classifiers_[0]
        return calibrated_classifier.predict_proba(X)
    
    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self)
        
        calibrated_classifier = self.calibrated_classifiers_[0]
        
        return self.classes_[np.argmax(calibrated_classifier.predict_proba(X), axis=1)]
    
   
    def save(self, fname):
        model_dict = {
                    'model' : self,
                    'features': self.features,
                    'n_features':len(self.features),
                    'n_training_examples' : len(self.X),
                    'hyperparameters' : self.best_params, 
                    'skew' : self.known_skew, 
                    }
        
        joblib.dump(model_dict, fname, compress=3)

    def _convert_param_grid(self, param_grid):
        """
        Converts a parameter grid to a hyperopt-friendly format if provide a list.
        """
        for p, values in param_grid.items():
            if isinstance(values, list): 
                return {p: hp.choice(p, values) for p,values in param_grid.items()}
            else: 
                return param_grid
            
    def _find_best_params(self,):
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
        self.best_params = space_eval(self.param_grid, best)

    def init_cv(self,):
        """Initialize the cross-validation generator"""
        # INITIALIZE MY CUSTOM CV SPLIT GENERATOR
        n_splits = self.cv_kwargs.get('n_splits')
        if self.cv == 'date_based':
            dates = self.cv_kwargs.get('dates', None)
            valid_size = self.cv_kwargs.get('valid_size', None)
            if dates is None:
                raise KeyError('When using cv = "date_based", must provide a date column in cv_kwargs')
            else:
                self.cv = DateBasedCV(n_splits=n_splits, dates=dates, y=self.y, valid_size=valid_size)
        elif self.cv =='kfolds':
            self.cv = KFold(n_splits=n_splits)
        else:
            self.cv = cv
        
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
        data =  [ast.literal_eval(df['params'].iloc[i]) for i in range(len(df))]
        results = []
        for p in params:
            df[p] = [data[i][p] for i in range(len(data))]
        df = df.drop(columns='params')

        df.to_pickle(fname)
        return df

    def objective(self, params):
        """Objective function for Hyperparameter Optimization"""
        # Keep track of evals
        self.ITERATION += 1
        start = timer()
        
        this_estimator = clone(self.pipe)       
        this_estimator.named_steps['model'].set_params(**params)

        parallel = Parallel(n_jobs=self.n_jobs)
        # Perform cross validation on the base estimator (no calibration!) 
        fit_estimators_ = parallel(delayed(
                _fit)(clone(this_estimator),self.X[train], self.y[train]) for train, _ in self.cv.split(self.X,self.y))
        
        # The fit estimators were fit on the training folds within clf
        scores = [self.scorer(model,self.X[test,:], self.y[test], **self.scorer_kwargs) 
                          for model, (_, test) in zip(fit_estimators_, self.cv.split(self.X, self.y))]

        run_time = timer() - start
        # Loss must be minimized (using NAUPDC as the metric!)
        loss = np.nanmean(scores)
        loss_variance = np.nanvar(scores, ddof=1)
        
        # Dictionary with information for evaluation
        if self.hyperparam_result_fname is not None:
            # Write to the csv file ('a' means append)
            of_connection = open(self.hyperparam_result_fname, 'a')
            self.writer = csv.writer(of_connection)
            self.writer.writerow([loss, loss_variance, params, self.ITERATION, run_time])

        return {'loss': loss, 'loss_variance': loss_variance, 'iteration': self.ITERATION, 'params' : params,
                'train_time': run_time, 'status': STATUS_OK}

    
class _CalibratedClassifier:
    """Probability calibration with isotonic regression or sigmoid.

    It assumes that base_estimator has already been fit, and trains the
    calibration on the input set of the fit function. Note that this class
    should not be used as an estimator directly. Use CalibratedClassifierCV
    with cv="prefit" instead.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. No default value since
        it has to be an already fitted estimator.

    method : 'sigmoid' | 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.

    classes : array-like, shape (n_classes,), optional
            Contains unique classes used to fit the base estimator.
            if None, then classes is extracted from the given target values
            in fit().

    See also
    --------
    CalibratedClassifierCV

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator, method='isotonic', classes=None):
        self.base_estimator = base_estimator
        self.method = method
        self.classes = classes

    def _preproc(self, X):
        n_classes = len(self.classes_)
        probabilities = self.base_estimator.predict_proba(X)[:,1]
        idx_pos_class = self.label_encoder_.\
            transform(self.base_estimator.classes_)

        return probabilities, idx_pos_class

    def fit(self, X, y):
        """Calibrate the fitted model

        Parameters
        ----------
        X : array-lie, shape (n_samples,)
            Predictions from the base_estimator

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.label_encoder_ = LabelEncoder()
        if self.classes is None:
            self.label_encoder_.fit(y)
        else:
            self.label_encoder_.fit(self.classes)

        self.classes_ = self.label_encoder_.classes_
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_.fit(X, y)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        probabilities, idx_pos_class = self._preproc(X)

        proba[:, 1] = self.calibrator_.predict(probabilities)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba



