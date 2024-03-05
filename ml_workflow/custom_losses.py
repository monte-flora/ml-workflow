import tensorflow as tf 
import tensorflow.keras.backend as K
from keras.losses import binary_crossentropy
#import tensorflow_probability as tfp

class MyWeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weights=[1.0, 1.0], name="custom_mse", **kwargs):
        super(MyWeightedMSE, self).__init__(name=name, **kwargs)
        self.w1 = weights[0]  # Unused in the current context but stored for potential future use
        self.w2 = weights[1]  # Unused in the current context but stored for potential future use

    def call(self, y_true, y_pred):
        # Penalties
        overprediction_penalty = 10.0
        # Increased penalty for underpredicting small hail sizes
        underprediction_penalty_small_hail = 15.0   
        underprediction_penalty_large_hail = 5.0 

        # Masks
        overpredict_mask = tf.logical_and(tf.greater(y_pred, y_true), tf.less_equal(y_true, 0.75))
        underpredict_mask_small_hail = tf.logical_and(tf.less(y_pred, y_true), tf.less_equal(y_true, 2.0))
        underpredict_mask_large_hail = tf.logical_and(tf.less(y_pred, y_true), tf.greater(y_true, 1.0))

        # Loss calculations
        overpredict_loss = tf.where(overpredict_mask, (y_pred - y_true) * overprediction_penalty, tf.zeros_like(y_true))
        underpredict_loss_small_hail = tf.where(underpredict_mask_small_hail, (y_true - y_pred) * underprediction_penalty_small_hail, tf.zeros_like(y_true))
        underpredict_loss_large_hail = tf.where(underpredict_mask_large_hail, (y_true - y_pred) * underprediction_penalty_large_hail, tf.zeros_like(y_true))

        # Combine loss components
        combined_loss = overpredict_loss + underpredict_loss_small_hail + underpredict_loss_large_hail

        # Return the mean loss
        return tf.reduce_mean(combined_loss)


class CustomHailSizeLoss(tf.keras.losses.Loss):
    def __init__(self, underpredict_weight=1.0, overpredict_weight=1.0, name="custom_hail_size_loss", **kwargs):
        super().__init__(name=name)
        self.underpredict_weight = underpredict_weight
        self.overpredict_weight = overpredict_weight

    def call(self, y_true, y_pred):
        # Create a mask for underpredictions where true hail size is less than 0.75
        underpredict_mask = tf.logical_and(tf.less(y_pred, y_true), tf.less(y_true, 0.75))

        # Create a mask for overpredictions where true hail size is greater than 1.0
        overpredict_mask = tf.logical_and(tf.greater(y_pred, y_true), tf.greater(y_true, 1.0))

        # Calculate the underprediction loss
        underpredict_loss = tf.where(underpredict_mask, 
                                     self.underpredict_weight * (y_true - y_pred), 
                                     tf.zeros_like(y_true))

        # Calculate the overprediction loss
        overpredict_loss = tf.where(overpredict_mask, 
                                    self.overpredict_weight * (y_pred - y_true), 
                                    tf.zeros_like(y_true))

        # Combine both losses
        combined_loss = tf.reduce_mean(underpredict_loss + overpredict_loss)
        return combined_loss


##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

# Fraction Skill Score (FSS) Loss Function - code taken from: https://github.com/CIRA-ML/custom_loss_functions
# Fraction Skill Score original paper: N.M. Roberts and H.W. Lean, "Scale-Selective Verification of Rainfall
#     Accumulation from High-Resolution Forecasts of Convective Events", Monthly Weather Review, 2008.
def make_fractions_skill_score(mask_size, num_dimensions, c=1.0, cutoff=0.5, want_hard_discretization=False):
    """
    Make fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.
    Parameters
    ----------
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    num_dimensions: int
        - Number of dimensions in the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.
    Returns
    -------
    fractions_skill_score: float
        - Fractions skill score.
    """

    pool_kwargs = {'pool_size': mask_size}
    if num_dimensions == 2:
        pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    elif num_dimensions == 3:
        pool1 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
    else:
        raise ValueError("Number of dimensions can only be 2 or 3")

    @tf.function()
    def fractions_skill_score(y_true, y_pred):
        """ Fractions skill score loss function """
        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)

        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    return fractions_skill_score

def csi(use_as_loss_function, use_soft_discretization,
            hard_discretization_threshold=None):
            
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            num_true_positives = K.sum(target_tensor * prediction_tensor)
            num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
            num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
            denominator = (
                num_true_positives + num_false_positives + num_false_negatives +
                K.epsilon()
            )

            csi_value = num_true_positives / denominator
            
            if use_as_loss_function:
                return 1. - csi_value
            else:
                return csi_value
        
        return loss

class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weights=[1.0, 1.0], threshold=0.5, name="WeightedMSE", **kwargs):
        super().__init__(name=name)
        self.threshold = threshold
        self.weight_above = weights[0]
        self.weight_below = weights[1]
        
    def call(self, y_true, y_pred):
        # Determine the weights based on the threshold
        weights = tf.where(y_true > self.threshold, self.weight_above, self.weight_below)
        
        # Calculate the squared differences
        squared_diff = tf.square(y_true - y_pred)
        
        # Apply weights
        weighted_squared_diff = weights * squared_diff
        
        # Return the mean of the weighted squared differences
        return tf.reduce_mean(weighted_squared_diff)

class RegressLogLoss_Normal(tf.keras.losses.Loss):
    """Regression log-loss that includes uncertainty via a 2-output regression setup.
    """
    
    def __init__(self,weights=[1.0,1.0],thresh=0.0,low=0,high=75, name="reglosloss_norm",
                 **kwargs):
        super().__init__()
        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.t = thresh
        self.low=low
        self.high=high

    def __str__(self):
        return ("RegressLogLoss_Normal()")

    def call(self, y_true, y_pred):
        
        
        y_true = tf.cast(y_true[...,0], tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
        mu = y_pred[...,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        
        dist = tfp.distributions.TruncatedNormal(loc=mu,scale=sigma,low=self.low,high=self.high)
        
        loss = -tf.math.log(dist.prob(y_true) + tf.keras.backend.epsilon())
        
        #build weight_matrix 
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,self.t),weights_for_nonzero,
                           weights_for_zero)
        
        #weight non-zero pixels more
        loss = tf.math.multiply(weight_matrix,loss)
        
        return tf.reduce_mean(loss)
    
class RegressLogLoss_SinhArcsinh(tf.keras.losses.Loss):
    """

    This is the negative log likelihood loss function for regression. 
    More specifically, this is an adaptation from the Barnes technique.

    This has weights that can be applied to prevent under-estimation of rare instance. 

    There is also a threshold value to apply to the weight to. 

    By default these parameters are set to 1 such that no weights are applied.

    """
    def __init__(self, weights=[1.0,1.0],thresh=0.0,name="reglogloss_sin",
                 **kwargs):
        super().__init__()
        #weights for weighted loss. w1 is > threshold weights. w2 is everything else. 
        self.w1 = weights[0]
        self.w2 = weights[1]
        #threshold to apply weights to  
        self.t = thresh

    def __str__(self):
        return ("RegressLogLoss_SinhArcsinh()")

    def call(self, y_true, y_pred):

        #force float64 to allow for big and small nums 
        y_true = tf.cast(y_true[...,0], tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        #Chase adaptation to prevent bad inital params 
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
        #these are the 4 parameters of the dist.
        mu = y_pred[...,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        gamma = y_pred[..., 2] 
        tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)
        
        #build dists.
        cond_dist = tfp.distributions.SinhArcsinh(mu, sigma,skewness=gamma,tailweight=tau)
        
        #Chase adaptation to prevent HUGE inital losses
        loss = -tf.math.log(cond_dist.prob(y_true)+tf.keras.backend.epsilon())
        
        #Chase adaptation to enable weighted loss function  
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,self.t),weights_for_nonzero,
                           weights_for_zero)
        loss = tf.math.multiply(weight_matrix,loss)
        
        #return loss 
        return tf.reduce_mean(loss)