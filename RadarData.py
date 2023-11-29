# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:13:23 2023

@author: opsuser
"""

import tensorflow as tf
import numpy as np

from datetime import datetime, timedelta

def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate <= endDate:
        yield currentDate
        currentDate += delta

# DATASET CLASS USED DURING TRAINING
class Dataset(tf.keras.utils.Sequence):
    
    def __init__(
            self, 
            dataset_dict,
            image_names,
            batch_size,
            leadtime_index,
            transform,
    ):
        self.leadtime_index = leadtime_index
        self.transform = transform
        self.valid_keys = image_names
        self.keys = sorted(list(dataset_dict.keys()))
        self.dataset = dataset_dict
        self.bs = batch_size

    def get_index(self,i):
   
        itime = self.valid_keys[i]
        index = self.keys.index(itime)
        
        # print(itime,index)
        
        x_indexes = list(range(index-6,index+1))
        y_index = index + self.leadtime_index
        # print(x_indexes,y_index)        
          
        xkey_list = [self.keys[i] for i in x_indexes]
        ykey = self.keys[y_index] 
        # print(xkey_list,ykey)
        
        
        # #(samples, frames, height, width, color_depth).               
        x = []
        for key in xkey_list:
            arr = np.array(self.dataset.get(key))
            x.append(arr)
        x = np.stack(x,0)
        
        y = np.array(self.dataset.get(ykey))            
        y = y[np.newaxis,:,:]
 
        x = data_preprocessing(x, self.transform)
        y = data_preprocessing(y, self.transform)
        
        return x,y
    
    def print_name(self,i):
        print(self.valid_keys[i])
    
    def __getitem__(self, index):
        
        X = []
        Y = []
        
        for i in range(index*self.bs,(index+1)*self.bs):
          x,y = self.get_index(i)
          X.append(x[np.newaxis,:])
          Y.append(y[np.newaxis,:])
        
        return np.concatenate(X,0),np.concatenate(Y,0)
        # return X,Y
        
    def __len__(self):
        return (len(self.valid_keys))//self.bs


# SET SCALER
# SHOULD PROBABLY EXPERIMENT WITH LOG TRANSFORM, WILL HELP REDUCE TO TENDENCY TO FORECAST 0MM

mean = 4.994233072440378
std = 14.027748696403338
vmax = 1601.0686
vmin = 1.063266e-12

def LogScaler(array):
    return np.log(array+0.01) 
    
def invLogScaler(array):
    return np.exp(array) - 0.01

def BoxCoxScaler(array, lmb = 0.1):
    return ((array)**lmb - 1) / lmb
    
def invBoxCoxScaler(array, lmb = 0.1):
    return np.power((array * lmb) + 1, 1 / lmb) - 1

def asinh_x_over_2(array):
    return np.arcsinh(array / 2)

def inverse_asinh_x_over_2(y):
    return 2 * np.sinh(y)
    
def data_preprocessing(X,method):
    X[X<0.1] = 0
    X = np.moveaxis(X, 0, -1)
    # X = X[::, ::, ::,np.newaxis]
    if method == 'log':
        X = LogScaler(X)
    elif method == 'boxcox':
        X = BoxCoxScaler(X)
    elif method == 'asinh':
        X = asinh_x_over_2(X)
    else:
        raise ValueError('Tranfornm method should be "log","boxcox" or "asinh"')
    return X

def data_postprocessing(nwcst,mask,fill_value,method):
    # 0. Squeeze empty dimensions
    nwcst = np.squeeze(np.array(nwcst))
    # 1. Convert back to rainfall depth
    if method == 'log':
        nwcst = invLogScaler(nwcst)
    elif method == 'boxcox':
        nwcst = invBoxCoxScaler(nwcst)
    elif method == 'asinh':
        nwcst = inverse_asinh_x_over_2(nwcst)
    else:
        raise ValueError('Tranfornm method should be "log","boxcox" or "asinh"')
    # 3. Return only positive values
    nwcst = np.where(nwcst>0.2, nwcst, 0)
    nwcst = np.ma.masked_where(mask,nwcst)
    nwcst.fill_value = fill_value
    return nwcst


# from scipy.stats import boxcox
# from scipy.special import inv_boxcox

# def boxcox_transform(x):
#     """Performs Box-Cox transformation on a given input array `x` and returns the transformed array and the optimal lambda"""
#     # find the optimal lambda
#     y, lambda_optimal = boxcox(x)
#     return y, lambda_optimal

# def boxcox_inverse(y, lambda_optimal):
#     """Performs the inverse of Box-Cox transformation on a given input array `y` and the optimal lambda"""
#     x = inv_boxcox(y, lambda_optimal)
#     return x
