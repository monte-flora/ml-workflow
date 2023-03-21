# A custom sklearn-based estimator that stacks
# multiple ML models and uses logistic regression
# to combine their predictions. 

from sklearnex import patch_sklearn
patch_sklearn()

# sklearn 
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.model_selection import cross_val_score

import pandas as pd 
import numpy as np 
import joblib 

from ml_workflow.tuned_estimator import TunedEstimator 
from sklearn.metrics import average_precision_score

def dates_to_groups(dates, n_splits=5): 
    """Separated different dates into a set of groups based on n_splits"""
    df = dates.copy()
    df = df.to_frame()
    
    unique_dates = np.unique(dates.values)
    np.random.shuffle(unique_dates)

    df['groups'] = np.zeros(len(dates))
    for i, group in enumerate(np.array_split(unique_dates, n_splits)):
        df.loc[dates.isin(group), 'groups'] = i+1 
        
    groups = df.groups.values
    
    return groups


class WeightedAverageClassifier(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    
    """
    Averaging the predictions from different estimators. 
    Weights are determined using K-folds cross-validation. 

    
    Parameters
    -------------------
    estimators: list of (str, estimator)
        Base prefit estimators which will be stacked together. Each element of the list is 
        defined as a tuple of string (i.e. name) and an estimator instance. 
        An estimator can be set to ‘drop’ using set_params.

        The type of estimator is generally expected to be a classifier. 
        However, one can pass a regressor for some use case (e.g. ordinal regression).
    

    Attributes
    ---------------
    model_ : The fit estimator (could be the original estimator, 
                a pipeline object, or a calibratedclassifier object)
    
    X_ : The training dataframe inputs 
    y_ : The target values 
    """
    def __init__(self, estimators, cv=None, scorer=average_precision_score, weights=None): 
        
        self.estimators = estimators 
        self.n_estimators = len(estimators)
        self.scorer = scorer 
        self.cv = cv
        if weights is None:
            n_est = len(estimators)
            weights = [1]*n_est
            weights /= n_est
            
        # this is only here for printing purposes
        self.weights = weights 
        
        self.weights_ = weights

        
    def get_score(self, X, y, groups, est):
        """Use cross-validation to determine proper weighing"""
        scores = [self.scorer(y[test_index], est.predict_proba(X.values[test_index,:])[:,1])
                     for _, test_index in self.cv] 
        
        return np.mean(scores)
        
    def _average_predictions(self, X):
        return np.average(np.array([est.predict_proba(X)[:,:] for est in self.estimators]),
                          weights=self.weights_,
                          axis=0)
    
        
    def fit(self, X, y, groups=None):
        """
        Determine the optimal weights for the different estimators 
        
        Parameters:
        ----------------
            X, pd.DataFrame, shape : (n_samples, n_features)
            y, shape: (n_samples, )
        
        """
        #if not isinstance(X, pd.DataFrame):
        #    X = pd.DataFrame(X)
            
        #self.features = X.columns
        
        # Rather than introducing two imputers, we can simply replace 
        # inf vals with nans. 
        #X.replace([np.inf, -np.inf], np.nan, inplace=True)

        #self.X_ = X 
        #self.y_ = y 
        
        if self.weights_ is None: 
        
            # For the moment, this just computes a simple average. 
            self.scores = [self.get_score(X, y, groups, est) for est in self.estimators]
            # Compute the total score and then base the 
            # the weights as the ratios of the total score. 
            self.total_score = np.sum(self.scores)
            
            self.weights_ = scores/total_score
       
        
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
        return self._average_predictions(X)
    
    def predict(self, X):
        """Model Predictions 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted values.
        """
        return self._average_predictions(X)   
    
    
    def save(self, fname): 
        data = {'model' : self,
                'features' : self.features,
               }
        joblib.dump(data, fname, compress=5)