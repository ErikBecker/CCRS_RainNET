# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:46 2023

@author: opsuser
"""



import os
import sys

import h5py
import random
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from RainNet import rainnet
from RadarData import Dataset

import numpy as np

# CLASS WILL OUTPUT MODEL WEIGHTS AFTER EVERY EPOCH
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
    print('\nEpoch %05d: saving optimizaer to %s' % (epoch + 1, filepath))


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
    
    DIR_DATA = "D:/data/OFFLINE"
    # DIR_DATA = "/scratch/nowcops/OFFLINE"
    
    training_index = 1
    transform_method = 'asinh'
    loss_function = 'rmse'

    # DEFINE DIRECTORIES
    INPUT_DATADIR = os.path.join(DIR_DATA, "ml_project", f"training_data_output_{transform_method}_T{str(training_index*5).zfill(3)}_{loss_function}")
    
    if not os.path.exists(INPUT_DATADIR):
        os.makedirs(INPUT_DATADIR)
    
    h5file = os.path.join(DIR_DATA, "ml_project", "rainrate_training_data.h5")
    imagefile = os.path.join(DIR_DATA, "ml_project",f"rainrate_training_keys_T+{str(training_index*5).zfill(3)}.txt")
      
    #RETURN LIST OF KEY NAMES
    image_names =open(imagefile,'r').read().split('\n')
      
    #SPLIT KEYS IN VARIOUS DATASETS
    test_images = [name for name in image_names if name[0:4]=="2021" or name[0:4]=="2022"]
    val_images = [name for name in image_names if name[0:4]=="2020"]
    train_images = [name for name in image_names if "202" not in name[0:3]]
    random.shuffle(train_images)        
    
    # OPENS H5 FILE FOR TRAINING
   
    h5f_dataset = h5py.File(h5file,"r")
   
    batch_size = 1
    max_epochs = 100

    # outdir = INPUT_DATADIR + "/training_cpt/"
    # run_desc = "test-train"
   
    train_dataset = Dataset(
                    dataset_dict=h5f_dataset,
                    image_names=train_images,
                    batch_size=batch_size,
                    leadtime_index=training_index,
                    transform=transform_method)
    
    # X,Y = train_dataset.__getitem__(20)
  
    # x = X[0,:,:,6]
    # y = Y[0,:,:]
        
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
    

    
    
    model = rainnet()
    # Load pretrained weights
    # model.load_weights(DATADIR + '/model_weights_rsme.h5')
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss="log_cosh")
   
    model_checkpoint_callback = MyModelCheckpoint(os.path.join(INPUT_DATADIR, 'model-{epoch:02d}-{val_loss:.4f}.hdf5'),monitor='val_loss',verbose=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,mode='min',restore_best_weights=True)
   
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999,clipnorm=1.0)
   
    model.compile(optimizer=optimizer,loss=custom_loss)
   
  
    history = model.fit(x=train_dataset,
                        validation_data=valid_dataset,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
   
#    # import matplotlib.pyplot as plt
   
#    # plt.plot(history.history['loss'], label='Training Loss')
#    # plt.plot(history.history['val_loss'], label='Validation Loss')
#    # plt.title('Model Loss')
#    # plt.xlabel('Epoch')
#    # plt.ylabel('Loss')
#    # plt.legend()
#    # plt.show()
   
   
    # model.save(INPUT_DATADIR + f'/model_weights_{transform_method}_{loss_function}_T+{str(training_index*5).zfill(3)}.h5')
    # tf.keras.models.save_model(model,INPUT_DATADIR + f'/model_state_{transform_method}_{loss_function}_T+{str(training_index*5).zfill(3)}.h5')
    # np.save(INPUT_DATADIR + f'/model_{transform_method}_{loss_function}_T+{str(training_index*5).zfill(3)}_history.npy',history)
    

