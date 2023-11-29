# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:15:04 2022

@author: opsuser
"""

import os
import sys

import time

import numpy as np

from numba import njit, prange, cuda
from datetime import datetime, timedelta

# SRCDIR = os.environ['SRCDIR']
# DATADIR = os.environ['DATADIR']
# MODE = os.environ['MODE']

# SRCDIR = "/home/ebecker/sf_work/scripts/MachLearn/RainNET"
# DATADIR = "/home/ebecker/sf_work/scripts/MachLearn/data"
# MODE = "OFFLINE"

SRCDIR = 'C:/Users/opsuser/OneDrive/work/scripts/MachLearn/RainNET'
DATADIR = 'C:/Users/opsuser/OneDrive/work/scripts/MachLearn/data'
# DATADIR = 'D:/data/OFFLINE/hdf5/comp'
# MODE = 'OFFLINE'

def getkeys(keysfile):
    return open(keysfile,'r').read().split('\n')

def find_valid_names_gpu(inputtimelist,idt):

    return_arr = np.ones(len(inputtimelist)) * -99.0
    d_in_ary = cuda.to_device(inputtimelist) 
    d_out_ary = cuda.to_device(return_arr)
    
    threadsperblock = 64
    blockspergrid = (return_arr.size + (threadsperblock - 1)) // threadsperblock

    #print(blockspergrid,threadsperblock)
    
    is_valid[blockspergrid,threadsperblock](d_in_ary,d_out_ary,idt)
   
    out_arr = d_out_ary.copy_to_host()
    
    return_arr = [inputtimelist[index] for index,item in enumerate(out_arr) if item==8]
   
    return return_arr


@cuda.jit
def is_valid(array_in,array_out,idt):
    
    i = cuda.grid(1)
    intime = array_in[i]
    bools = 0
    for j in range(-6,2): #VALID IMAGE T-30MIN TO T+5MIN 
    
        if j == 1:
            t = 300 * idt
        else:
            t = 300
        checktime = intime + t * j    
        for looptime in array_in:
            if looptime==checktime:
                bools += 1
                break
      
    array_out[i] = bools

def round_datetime(dt):
    total_seconds = dt.minute * 60 + dt.second
    seconds_past_last_five = total_seconds % 300
    # If the seconds are in the first half of the 5-minute interval, round down; otherwise, round up
    if seconds_past_last_five < 150:
        dt = dt - timedelta(seconds=seconds_past_last_five)
    else:
        dt = dt + timedelta(seconds=(300 - seconds_past_last_five))
    return dt

def floor_datetime(dt, delta=timedelta(minutes=5)):
    return (dt + (datetime.min - dt) % delta) - delta
   
if __name__ == "__main__":
    
    DIR_DATA = "D:/data/OFFLINE"
    # DIR_DATA = "/scratch/nowcops/OFFLINE"

    h5file = os.path.join(DIR_DATA, "ml_project", "rainrate_training_data.h5")
    txtfile = os.path.join(DIR_DATA, "ml_project", "rainrate_training_key.txt")
      
    h5keys = sorted(getkeys(txtfile))

    start = time.perf_counter()       
    datetime_keys = {round_datetime(datetime.strptime(x, "%Y%m%d%H%M%S")):x for x in h5keys}
    sec_keys = np.array([x.timestamp() for x in list(datetime_keys.keys())])
    gpu_image_names = {}
    for i in range(1,19):
        gpu_image_names[i] = find_valid_names_gpu(sec_keys,i)
        end = time.perf_counter()
        print(f"Iter = {i} Time at out = {(end - start)}s")
        
    # image_names = [datetime.fromtimestamp(x).strftime("%Y%m%d%H%M%S") for x in gpu_image_names]
    end = time.perf_counter()
    print("GPU Elapsed Time  = {}s".format((end - start)))
    
    for i in range(1,19):
        
        key_names = [datetime.fromtimestamp(x) for x in  gpu_image_names[i]]
        image_names = [datetime_keys[x] for x in key_names]
        
        imagefile = os.path.join(DIR_DATA, "ml_project",f"rainrate_training_keys_T+{str(i*5).zfill(3)}.txt")
    
        tfile = open(imagefile,'w')
        tfile.writelines('\n'.join(image_names))
        tfile.close()
    
    
    
    
# @njit
# def find_valid_names_cpu(inputtimelist):
       
#     out_arr = np.ones(len(inputtimelist)) * -99.0

#     for i in prange(0,len(inputtimelist)):
    
#         intime = inputtimelist[i]    
#         bools = np.empty(8)
#         for j in range(-6,2):
#             checktime = intime + 300 * j           
#             bools[j+6] = checktime in inputtimelist
            
#         if (np.all(bools)): out_arr[i] = intime
        
#     return_arr = [inputtimelist[index] for index,item in enumerate(out_arr) if item!=-99.]
    
#     return return_arr


# @cuda.jit
# def is_valid(array_in,array_out):
    
#      i = cuda.grid(1)
#      intime = array_in[i]
#      bools = 0
#      for j in range(-6,2):
#          checktime = intime + 300 * j    
#          for looptime in array_in:
#              if looptime==checktime:
#                  bools += 1
#                  break
        
#      array_out[i] = bools

# def find_valid_names_gpu(inputtimelist):

#     return_arr = np.ones(len(inputtimelist)) * -99.0
#     d_in_ary = cuda.to_device(inputtimelist) 
#     d_out_ary = cuda.to_device(return_arr)
    
#     threadsperblock = 64
#     blockspergrid = (return_arr.size + (threadsperblock - 1)) // threadsperblock

#     #print(blockspergrid,threadsperblock)
    
#     is_valid[blockspergrid,threadsperblock](d_in_ary,d_out_ary)
   
#     out_arr = d_out_ary.copy_to_host()
    
#     return_arr = [inputtimelist[index] for index,item in enumerate(out_arr) if item==8]
   
#     return return_arr

# GPU Elapsed Time  = 15.109234600000491s
# CPU Elapsed Time  = 865.2784734000006s
# Python Elapsed Time  = 12273.3882654s
    
    