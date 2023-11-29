# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:33:49 2023

@author: opsuser
"""

import os
import h5py
import multiprocessing

import xarray as xr
import numpy as np

from glob import glob
from datetime import datetime

def return_directory_files(directory_path,extension):
    
    filelist = []
    
    if os.path.exists(directory_path):
    
        radar_id = os.listdir(directory_path)
        
        directories = {}
        for radar in radar_id:
            subdirectories = sorted(os.listdir(os.path.join(directory_path,radar)))
            directories[radar] = [x for x in subdirectories if os.path.isdir(os.path.join(directory_path,radar,x))]
            
        for radar in directories:
            for date in directories[radar]:
                search_path = os.path.join(directory_path,radar,date,f"*.{extension}")
                filelist.append(glob(search_path))
                
        filelist = [item for sublist in filelist for item in sublist]
    
    return filelist

def process_file(infile, h5f):
    
    print(infile)

    xrate = xr.open_dataarray(infile)
    mask = xrate < 0
    xrate = xrate.where(~mask, 0.)
    xrate = xrate.fillna(-1)
    rate = np.flipud(xrate.values)
    stime = datetime.strptime(xrate.start_time, "%Y-%m-%d %H:%M:%S")
    stimestr = stime.strftime("%Y%m%d%H%M%S")
    
    with h5f.lock:
        dataset = h5f.create_dataset(stimestr, data=rate)

    return stimestr

if __name__ == "__main__":
    DIR_DATA = "D:/data/OFFLINE"
    # DIR_DATA = "/scratch/nowcops/OFFLINE"

    inputFiles = sorted(return_directory_files(os.path.join(DIR_DATA, "qpe", "rainrate"), "nc"))
    inputFiles = [infile for infile in inputFiles if int(os.path.basename(infile).split('.')[0]) == 200 and 'rainrate-adj' in os.path.basename(infile)]

    h5file = os.path.join(DIR_DATA, "ml_project", "rainrate_training_data.h5")
    txtfile = os.path.join(DIR_DATA, "ml_project", "rainrate_training_key.txt")

    if not os.path.exists(os.path.dirname(h5file)):
        os.makedirs(os.path.dirname(h5file), exist_ok=True)

    with h5py.File(h5file, 'w') as hdf5_file:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cores)
        timekeys = pool.map(lambda ifile: process_file(ifile, hdf5_file), sorted(inputFiles))

    with open(txtfile, "w") as text_file:
        for key in timekeys:
            text_file.write(key + "\n")
          
    
    
    