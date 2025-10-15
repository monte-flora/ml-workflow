#imports 
import os.path
import random
import shutil
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import xarray as xr 
import pandas as pd
from tensorboard.plugins.hparams import api as hp
import sys
import matplotlib.pyplot as plt
import io
import glob
import gc 
from pathlib import Path
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class KerasRandomSearch:
    def __init__(self, model_fn, X_train, y_train, X_val, y_val, 
                 search_space, 
                 n_iter=100, 
                 logdir = '/work/mflora/ML_DATA/NN_MODELS', 
                 epoches=20, patience=7, 
                 fit_verbose=0, pre_fit=False, 
                 tensorboard_metrics=['loss', 'rmse', 'crmse', 'csi_1in']):
        
        self.model_fn = model_fn
        self.search_space = search_space
        self.n_iter = n_iter
        self.logdir = logdir
        self.epoches = epoches 
        self.VAL_BATCH_SIZE = 1024*5
        self.TRAIN_BATCH_SIZE = 2048*5
        self._declare_tensorboard_metrics(tensorboard_metrics)
        self.patience = patience
        self.fit_verbose = fit_verbose
        self.pre_fit = pre_fit
        self.data = self.to_tf_dataset(X_train, y_train, X_val, y_val)
    
    def _declare_tensorboard_metrics(self, metrics):
        """This is all the metrics you want in your tensorboard"""
        self.METRICS = []
        for group in ['train', 'validation']:
            for m in metrics:
                display_name = f"{m} (val.)" if group == 'validation' else f"{m} ({group})"
                met = hp.Metric(f"epoch_{m}", group=group, display_name=display_name)
                self.METRICS.append(met)


    def _fit_once(self, hparams, session_id, cache=None):
        """Run a training/validation session.
        Flags must have been parsed for this function to behave.
        Args:
          session_id: A unique string ID for this session.
          hparams: A dict mapping hyperparameters in `HPARAMS` to values.
        """
        model = self.model_fn(self.input_shape, hparams=hparams, iter_id=session_id)

        logdir = os.path.join(self.logdir, str(session_id))

        ds_train, ds_val = self.data

        #cache data 
        if cache is None:
            pass
        else:
            print('putting data into local cache!')
            ds_train = ds_train.cache(cache + 'training/')
            ds_val = ds_val.cache(cache + 'validation/')

        #batch the training data accordingly 
        buffer_size = ds_train.cardinality().numpy()
    
        ds_train = ds_train.shuffle(
            buffer_size=buffer_size).batch(self.TRAIN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        #this batch is arbitrary, just needed so that you dont overwelm RAM. 
        ds_val = ds_val.batch(self.VAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        callback = tf.keras.callbacks.TensorBoard(
            logdir,
            update_freq='epoch',
            profile_batch=0,  # workaround for issue #2084
        )
        
        hparams_callback = hp.KerasCallback(logdir, hparams)

        callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=self.patience, 
                                                   restore_best_weights=True)
    
        #should kill it if there is an especially bad training loss... (i.e., nan; no need to waste time) 
        nan_callback = tf.keras.callbacks.TerminateOnNaN()

        #add images to board 
        #print(model.summary())
        result = model.fit(ds_train,
            epochs=self.epoches,
            shuffle=False,
            validation_data=ds_val,
            callbacks=[callback, hparams_callback, callback_es,nan_callback],
                           verbose=self.fit_verbose)

        #save trained model, need to build path first 
        model.save(Path(self.logdir).joinpath(f'model_{session_id}.h5'))

        #do some cleanup
        del model, ds_train,ds_val

        gc.collect()
    
    def fit_preprocessors(self, X):
        """Impute missing data and scale"""
        names = ['imputer', 'scaler']
        
        preprocessors = [ SimpleImputer(strategy='mean'), 
                          MinMaxScaler(),
                        ]
        self.preprocessors = [p.fit(X) for p in preprocessors]
        
        for n, p in zip(names, self.preprocessors):
            path = os.path.join(self.logdir, f"{n}.joblib")
            print(f'Saving {n} to {path} ...')
            joblib.dump(p, path)
        
    def apply_preprocessing(self, X): 
        """Apply preprocessing transformations"""
        X_trans = X.copy()
    
        for p in self.preprocessors:
            X_trans = p.transform(X_trans)
    
        return X_trans
    
    def to_tf_dataset(self, X_train, y_train, X_val, y_val):
        """Fit and transform data using the preprocessors, then 
        convert to tensorflow datasets."""
        self.input_shape = (X_train.shape[1],)
        
        self.fit_preprocessors(X_train)
        
        X_train = self.apply_preprocessing(X_train)
        X_val = self.apply_preprocessing(X_val)
        
        if hasattr(y_train, 'values'):
            y_train = y_train.values
            
        if hasattr(y_val, 'values'):
            y_val = y_val.values 
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_dataset =  tf.data.Dataset.from_tensor_slices((X_val, y_val))

        data = (train_dataset, valid_dataset)
    
        return data
    
    def fit(self, verbose=False, cache=None, seed=123):
        """Perform random search over the hyperparameter space.
        Arguments:
          logdir: The top-level directory into which to write data. This
            directory should be empty or nonexistent.
          verbose: If true, print out each run's name as it begins.
        """        
        rng = random.Random(seed) #changed this seed to get a new param set  

        with tf.summary.create_file_writer(self.logdir).as_default():
            hparams_list = [h for arg, h in self.search_space.items()]
            hp.hparams_config(
                hparams=hparams_list, 
                metrics=self.METRICS)

        for session_id in range(1, self.n_iter+1):
            session_id = str(session_id)
            
            hparams = {arg: h.domain.sample_uniform(rng) for arg, h in self.search_space.items()}
            hparams_string = str(hparams)

            if verbose:
                print(f"---Running training session {session_id}---")
                print(f'Hyperparameters: \n {hparams_string}')
                        
            # Clear previous models and metrics
            print('Clear previous session....') 
            tf.keras.backend.clear_session()                    
                    
            self._fit_once(
                        session_id=session_id,
                        hparams=hparams,
                        cache=cache,
                    )
