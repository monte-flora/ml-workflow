#from scikeras.wrappers import KerasClassifier, KerasRegressor
from typing import Dict, Iterable, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import (Dense, 
                                     Activation, 
                                     Conv2D, 
                                     Conv3D,  
                                     Input, 
                                     AveragePooling2D, 
                                     AveragePooling3D, 
                                     Flatten, 
                                     LeakyReLU
                                    )
from tensorflow.keras.layers import (Dropout, BatchNormalization, 
                                    ELU, MaxPooling2D, MaxPooling3D, ActivityRegularization)
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras.metrics import RootMeanSquaredError, AUC#, F1Score

from .custom_losses import (RegressLogLoss_Normal, 
                            RegressLogLoss_SinhArcsinh, 
                            WeightedMSE, 
                            CustomHailSizeLoss,
                            MyWeightedMSE
                           )

from .custom_metrics import (ParaRootMeanSquaredError2, 
                             ConditionalRootMeanSquaredError, 
                             ConditionalParaRootMeanSquaredError, 
                             CSIScoreThreshold
                            )


# The custom classifier 
import sys
sys.path.insert(0, '/home/monte.flora/python_packages/deep-severe')

from dpsevere.models.keras_builder import TFPreProcessPipeline

from tensorflow.keras.metrics import MeanSquaredError


class KerasTabular():

    _CUSTOM_LOSSES = ['WeightedMSE', 'CustomHailSizeLoss', 'MyWeightedMSE']
    
    def __init__(
        self,
        input_shape, 
        mode = 'regression', 
        output_size = 1, 
        initial_hidden_layer_size =100,
        optimizer="adam",
        loss='mse', 
        optimizer__learning_rate=0.0001,
        layer_size_decay_rate = 0.75,
        num_layers = 3, 
        activation='leaky_relu',
        batch_norm = True,
        l1_weight=0.0,
        l2_weight=0.0, 
        dropout_rate=0.1,
        weight = 1.0,
        thresh = 0.0, 
        underpredict_weight = 1.0, 
        overpredict_weight = 1.0,
        epochs=200,
        verbose=1,
        metrics = [],
        final_activation=None,
        use_multiple_gpus=False,
        iter_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_layer_sizes = self._hidden_layer_sizes(initial_hidden_layer_size, 
                                                           num_layers, layer_size_decay_rate)
        self.mode = mode
        self.output_size = output_size
        self.input_shape = input_shape
        self.batch_norm = batch_norm
        self.activation = activation 
        self.optimizer = optimizer
        self.optimizer__learning_rate = optimizer__learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight 
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.clipnorm = 1 
        self.metrics = metrics
        self.final_activation = final_activation
        self.weight = weight
        self.thresh = thresh 
        self.underpredict_weight = underpredict_weight 
        self.overpredict_weight = overpredict_weight
        self.iter_id = iter_id
        
        # If using multiple GPUs, then need to build the model
        # inside the MirroredStrategy 
        if use_multiple_gpus:
            print('Building for multiple GPUs...')
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = self.build()
        else:
            print('Building for a single GPU...')
            self.model = self.build()
        
    def _get_optimizer(self):
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.optimizer__learning_rate, clipnorm=self.clipnorm)
        elif self.optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=self.optimizer__learning_rate)
        
        return opt
        
    def _hidden_layer_sizes(self, initial_size, num_layers, decay_rate):
        """
        Calculate hidden layer sizes using exponential decay.
    
        :param initial_size: Size of the first hidden layer.
        :param num_layers: Total number of hidden layers.
        :param decay_rate: Rate of decay for layer sizes.
        :return: List of sizes for each hidden layer.
        """
        return (int(initial_size * np.exp(-decay_rate * i)) for i in range(num_layers))

    def _get_regularization_layer(self,  l1_weight, l2_weight ):
        """ Creates a regularization object.
        """
        return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)
    
    def _get_activation_layer(self, function_name, alpha_parameter=0.2): 
        """ Creates an activation layer. 
        :param function name: Name of activation function (must be accepted by
                        `_check_activation_function`).
        :param alpha_parameter: Slope (used only for eLU and leaky ReLU functions).
        :return: layer_object: Instance of `keras.layers.Activation`,
                        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
        """
        if function_name == 'elu': 
            return ELU( alpha = alpha_parameter )
        if function_name == 'leaky_relu': 
            return LeakyReLU( alpha = alpha_parameter) 
        return Activation(function_name)  
    
    def _get_dropout_layer(self):
        """ Create a dropout object for the dense layers
        """
        return Dropout( rate = self.dropout_rate )
    
    def _get_batch_norm_layer( self ):
            """Creates batch-normalization layer.

            :return: layer_object: Instance of `keras.layers.BatchNormalization`.
            """
            # TODO: Ask ChatGPT! 
            #return tf.keras.layers.Lambda(lambda x: x * scale)
            
            return BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    
    def _get_dense_layer(self, num_neurons, l1_weight, l2_weight, dense_bias='zeros',
                        kernel_initializer = 'glorot_uniform', output_layer=False): 
        """ Create a Dense layer with optionally regularization. 
        """
        return Dense( num_neurons ,
                              kernel_initializer = kernel_initializer,
                              use_bias           = True,
                              bias_initializer   = dense_bias,
                              activation         = None,
                              kernel_regularizer = self._get_regularization_layer( l1_weight, l2_weight) )
    
    def _get_loss(self, loss_name, weight=1.0, thresh=0.0, 
                  underpredict_weight=1.0, overpredict_weight=1.0):
        
        
        if loss_name in self._CUSTOM_LOSSES:
            if loss_name == 'WeightedMSE':
                print(f"{weight=}")
                return WeightedMSE(weights=[weight, 1.0])
            
            elif loss_name == 'CustomHailSizeLoss':
                return CustomHailSizeLoss(underpredict_weight=underpredict_weight, 
                                          overpredict_weight=overpredict_weight)
            
            elif loss_name == 'MyWeightedMSE':
                return MyWeightedMSE(weights=[1.0, 1.0])
            
            elif loss_name == 'RegressLogLoss_SinhArcsinh':
                return RegressLogLoss_SinhArcsinh(weights=[weight,1.0],thresh=thresh)
            
        else:
            return loss_name
    
    
    def build(self):
        
        self.model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.input_shape))
        self.model.add(inp)
        
        # add the scaling. 
        #scaler_layer = TFPreProcessPipeline(scaler='minmax', axes=(1,0),
        #                                    scale=self.scale, offset=self.offset)(inp)
        #self.model.add(scaler_layer)
        
        for n in self.hidden_layer_sizes:
            # Apply Dense layer with regularization 
            dense_layer = self._get_dense_layer(n, self.l1_weight, self.l2_weight)
            self.model.add(dense_layer)
            
            # Apply activation 
            activation_layer = self._get_activation_layer( 
                    function_name = self.activation )
            self.model.add(activation_layer)
        
            # Apply batch normalization (optional) 
            if self.batch_norm:
                batch_norm_layer= self._get_batch_norm_layer()
                self.model.add(batch_norm_layer)
            
            if self.dropout_rate > 0 : 
                dropout_layer = self._get_dropout_layer()
                self.model.add(dropout_layer)
            
        # Add the final layer. 
        final_activation = None if self.mode == 'regression' else 'sigmoid'
        
        if self.final_activation:
            final_activation = self.final_activation
        
        out = keras.layers.Dense(self.output_size, final_activation)
        self.model.add(out)
        
        # Compile the model.
        optimizer = self._get_optimizer()

        loss = self._get_loss(self.loss, weight=self.weight, thresh=self.thresh, 
                                underpredict_weight=self.underpredict_weight, 
                                overpredict_weight=self.overpredict_weight)
            
        self.model.compile(optimizer =optimizer, 
                           loss = loss, 
                           metrics = self.metrics
                          )
                
        if self.verbose > 0:
            print(self.model.summary())

        return self.model
    
    def fit(self, X_train, X_val, y_train=None, y_val=None):
        """
        Fit the estimator 
        
        Parameters:
        ----------------
            X_train, array, shape : (n_samples, ny, nx, n_features)
            y_train, shape: (n_samples, )
        """
        # Scale the data before fitting. 
        #if self.scaler is not None:
        #    X_train = self.scaler.transform(X_train)
        #    X_val= self.scaler.transform(X_val)
        
        #X_train, y_train = self.to_tensor(X_train, y_train)
        #X_val, y_val = self.to_tensor(X_val, y_val)
        callbacks =  [keras.callbacks.EarlyStopping(
                                monitor='val_prc',
                                patience=10,
                                restore_best_weights=True,
                          ),
                         keras.callbacks.ModelCheckpoint(
                             filepath = 'my_model.h5',
                             monitor = 'val_prc',
                             save_best_only=True,
                         ),
                        ]
        if y_train is None:
            self.conv_hist = self.conv_model_.fit(x=X_train, 
                           validation_data = X_val,
                           epochs = 50,
                           callbacks=callbacks,
                           class_weight=self.class_weights, 
                          )
            
        else:
            self.conv_hist = self.conv_model_.fit(X_train, y_train, 
                           validation_data = (X_val, y_val),
                           epochs = 50,
                           callbacks=callbacks,
                           class_weight=self.class_weights, 
                          )
        
        if self.tuning:
            del self.conv_model_, X_train, X_val
            gc.collect()

            cuda.select_device(0)
            cuda.close()
    
    def load(self, fname):
        """Loads the Keras and Scaler models"""
        self.conv_model_ = keras.models.load_model(fname, 
                             custom_objects = 
                              {'brier_skill_score_keras' : brier_skill_score_keras})
    
        self.scaler = joblib.load(fname.replace('.h5', '_scaler.h5'))
    
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
        proba = np.zeros((X.shape[0], 2))

        X_scaled = self.scaler.transform(X)
        proba[:, 1] = self.conv_model_.predict(X_scaled)[:,0]
        proba[:, 0] = 1. - proba[:, 1]

        return proba
        
    def predict(self, X):
        """Model Predictions 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, )
            The predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.conv_model_.predict(X_scaled)[:,0]
    
    def save(self, fname='classifier.h5'):
        # Save the model. 
        self.conv_model_.save(fname)
        joblib.dump(self.scaler, fname.replace('.h5', '_scaler.h5'), compress=3)
        
        # Save the conv_hist results. 
        
    def plot_train_loss(self, add_validation=True, metric='loss'):
        f, ax = plt.subplots(dpi=150, figsize=(6,4))

        ax.plot(self.conv_hist.history[metric], label=f'Training {metric.title()}', )
        if add_validation: 
            ax.plot(self.conv_hist.history[f'val_{metric}'], label=f'Validation {metric.title()}', 
            )
        ax.grid(alpha=0.5, ls='dashed')
        ax.legend()
        ax.set_ylabel(f'{metric.title()}')
        ax.set_xlabel('Epoches')
        
        return f, ax