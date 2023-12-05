# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:46 2023

@author: opsuser
"""

import os
import glob
import sys
from time import time, strftime, localtime

import h5py
import random
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from RainNet import rainnet
from RadarData import Dataset

import numpy as np


# Sub-class which saves the optimizer state every epoch (redundant since full model saved)
class MyModelCheckpoint(ModelCheckpoint):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch,logs)\

    # Also save the optimizer state
    filepath = self._get_file_path(epoch=epoch,batch=1,logs=logs)
    filepath = filepath.rsplit( ".", 1 )[ 0 ] 
    filepath += ".pkl"

    with open(filepath, 'wb') as fp:
      pickle.dump(
        {
          'opt': model.optimizer.get_config(),
          'epoch': epoch+1
         # Add additional keys if you need to store more values
        }, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('\nEpoch %05d: saving optimizer to %s' % (epoch + 1, filepath))


# Sub-class for saving model only every x frequency epochs if time consuming to save model
class EpochModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 filepath,
                 frequency=1,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 options=None,
                 **kwargs):
        super(EpochModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                   mode, "epoch", options)
        self.epochs_since_last_save = 0
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.epochs_since_last_save % self.frequency == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass


def custom_loss(y_true, y_pred):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    loss = rmse          
    return loss

#import tensorflow as tf
#import tensorflow_probability as tfp
#
#def pearson_correlation_loss(y_true, y_pred):
#    pearson_correlation = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
#    # We subtract the correlation from 1 because we want our loss to be small when correlation is high
#    return 1 - pearson_correlation
#
#import tensorflow as tf
#import tensorflow_probability as tfp
#
## Assume y_true and y_pred are one-dimensional tensors
#def quantile_mapping(y_true, y_pred, num_quantiles=100):
#    # Step 1: Compute the quantiles
#    y_true_quantiles = tfp.stats.quantiles(y_true, num_quantiles=num_quantiles)
#    y_pred_quantiles = tfp.stats.quantiles(y_pred, num_quantiles=num_quantiles)
#
#    # Step 2: Find corresponding values in y_pred for each quantile
#    y_pred_sorted = tf.sort(y_pred)
#    y_pred_values = tf.gather(y_pred_sorted, tf.cast(y_pred_quantiles * tf.size(y_pred, out_type=tf.float32), tf.int32))
#
#    # Step 3: Replace values in y_pred with corresponding values from y_true
#    y_pred_mapped = tfp.math.interp_regular_1d_grid(y_pred, y_pred_values, y_true_quantiles)
#
#    return y_pred_mapped

if __name__== "__main__":

    print('Start', strftime('%Y-%m-%d %H:%M:%S', localtime(time())))
    DIR_DATA = "/data/projects/17001770/weather_department/nwp/leeck/port_aspire2a/"
    
    training_index = 6
    #transform_method = 'log'
    transform_method = 'boxcox'
    #transform_method = 'asinh'
    loss_function = 'rmse'

    # Define directories (new one created if it does not exist)
    INPUT_DATADIR = os.path.join(DIR_DATA, "ml_project", f"training_data_output_{transform_method}_T{str(training_index*5).zfill(3)}_{loss_function}")
    
    if not os.path.exists(INPUT_DATADIR):
        os.makedirs(INPUT_DATADIR)
    
    h5file = os.path.join(DIR_DATA, "ml_project", "rainrate_training_data.h5")
    imagefile = os.path.join(DIR_DATA, "ml_project", f"rainrate_training_keys_T+{str(training_index*5).zfill(3)}.txt")

    print(h5file)
    print(imagefile)
      
    # Return list of key names
    image_names =open(imagefile,'r').read().split('\n')
      
    # Split keys in various datasets
    test_images = [name for name in image_names if name[0:4]=="2021" or name[0:4]=="2022"]
    val_images = [name for name in image_names if name[0:4]=="2020"]
    train_images = [name for name in image_names if "202" not in name[0:3]]
    random.shuffle(train_images)    
    
    # Open H5 file for training
    h5f_dataset = h5py.File(h5file, "r")
   
    batch_size = 80
    max_epochs = 50 #100

    # Split dataset
    train_dataset = Dataset(
                    dataset_dict=h5f_dataset,
                    image_names=train_images,
                    batch_size=batch_size,
                    leadtime_index=training_index,
                    transform=transform_method)
        
    valid_dataset = Dataset(
                    dataset_dict=h5f_dataset,
                    image_names=val_images,
                    batch_size=batch_size,
                    leadtime_index=training_index,
                    transform=transform_method)
       
    test_dataset =  Dataset(
                    dataset_dict=h5f_dataset,
                    image_names=test_images,
                    batch_size=batch_size,
                    leadtime_index=training_index,
                    transform=transform_method)
    
    print('Before parallelising', strftime('%Y-%m-%d %H:%M:%S', localtime(time())))

    # Parallelise to train on multiple GPUs. Otherwise just remove strategy and strategy.scope()
    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        model = rainnet()

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

        # Check if checkpoints exist, if not start new training otherwise continue from last checkpoint.
        ckpt_files = glob.glob(INPUT_DATADIR + '/model-*')
        ckpt_files.sort()
        if ckpt_files == []:
            print("No existing checkpoints, starting cold start training")
            model.compile(optimizer, loss=custom_loss)
            latest_epoch = 0 
        else:
            latest = ckpt_files[-1]
            print("Loading latest checkpoint from: ", latest)

            model = tf.keras.models.load_model(latest, custom_objects={'custom_loss': custom_loss})
            latest_epoch = int(latest.split('/')[-1].split('-')[1])

    print('Before callback', strftime('%Y-%m-%d %H:%M:%S', localtime(time())))

    model_checkpoint_callback = ModelCheckpoint(os.path.join(INPUT_DATADIR, 'model-{epoch:02d}-{val_loss:.4f}.hdf5'), monitor='val_loss', verbose=1)
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
  
    print('Before fit', strftime('%Y-%m-%d %H:%M:%S', localtime(time())))
    print('Starting training from epoch', str(latest_epoch))
    history = model.fit(x=train_dataset,
                        validation_data=valid_dataset,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        initial_epoch=latest_epoch,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
   
    # Final re-save if early stopping triggered and job ends early, otherwise wallclock usually kill jobs early
    model.save(INPUT_DATADIR + f'/model_{transform_method}_{loss_function}_T+{str(training_index*5).zfill(3)}.h5')
