from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class TensorFlowPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, tf_model, preprocessors):
        self.tf_model = tf_model
        self.preprocessors = preprocessors

    def fit(self, X, y=None):
        # No fitting needed as model is already trained
        return self

    def transform(self, X): 
        """Apply preprocessing transformations"""
        X_trans = X.copy()
    
        for p in self.preprocessors:
            X_trans = p.transform(X_trans)
    
        return X_trans
    
    def predict(self, X):
        # Use the model for prediction
        X_trans = self.transform(X)
        
        return self.tf_model.predict(X_trans)
    
    
# Step 1: Create a custom wrapper for the TensorFlow model
class TensorFlowModelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        # No fitting needed as model is already trained
        return self

    def predict(self, X):
        # Use the model for prediction
        return self.model.predict(X)
    
    
# Step 1: Create a custom wrapper for the TensorFlow model
class HailClassifierRegressorModel(BaseEstimator, TransformerMixin):
    def __init__(self, small_reg_model, large_reg_model, class_model, 
                 class_sig_model=None, massive_reg_model=None, prob_threshold=0.3):
        self.small_reg_model = small_reg_model
        self.large_reg_model = large_reg_model
        self.massive_reg_model = massive_reg_model
        
        self.class_model = class_model
        self.class_sig_model = class_sig_model
        self.prob_threshold = prob_threshold

    def fit(self, X, y=None):
        # No fitting needed as model is already trained
        return self

    def predict(self, X):
        # Use the model for prediction
        
        final_preds = np.zeros(len(X))
        
        class_preds = self.class_model.predict_proba(X)[:,1]
        if self.class_sig_model:
            class_sig_preds = self.class_sig_model.predict_proba(X)[:,1]
        
        
        small_model_inds = class_preds<=self.prob_threshold
        large_model_inds = class_preds>self.prob_threshold
        
        massive_inds = class_sig_preds>=0.15
        
        #small_model_pred = X['hailcast__time_max__amp_ens_mean_spatial_perc_90'].values
        
        final_preds[large_model_inds] = self.large_reg_model.predict(X[large_model_inds])[:,0]
        final_preds[small_model_inds] = self.small_reg_model.predict(X[small_model_inds])[:,0]
        
        final_preds[massive_inds] = self.massive_reg_model.predict(X[massive_inds])[:,0]
        
        #final_preds[small_model_inds] = small_model_pred[small_model_inds]
        
        return final_preds
