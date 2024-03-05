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


class AnySevereClassifier(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    
    """
    Combine the individual hazard classifiers into a single probability 
    
    P(Severe) = 1 - [(1-P(Hail)) * (1-P(Wind)) * (1-P(Torn))]

    
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
    def __init__(self, estimators): 
        
        self.estimators = estimators 
        self.n_estimators = len(estimators)
  
    def _combine(self, X):
        probs = np.zeros((len(X), 2))
        results = [1.0 - est.predict_proba(X)[:,1] for est in self.estimators]
        probs[:,1] =  1.0 - np.product(results, axis=0)
        probs[:,0] = 1.0 - probs[:,1]
        
        return probs 
            
        
    def fit(self, X, y, groups=None):
        """No fitting. """
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
        return self._combine(X)
    
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
        return self._combine(X)   
    
    
    def save(self, fname): 
        data = {'model' : self,
                'features' : self.features,
               }
        joblib.dump(data, fname, compress=5)