# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:02:06 2022

compile shock tube data into simple file for reading in later

@author: scott
"""


import numpy as np
import matplotlib.pyplot as plt

save_path = r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\\"


which_set = ['1ArHe', '2ArHe', '3ArHe', '4ArHe',
             '1ArL', # with bad lasers (?)
             '1Ar', '2Ar', '3Ar', '4Ar', 
             'vac21', 'vac23']

# which_files = [[0,1], [2], [3], [4],
#                [5,6],
#                [10,11], [7], [8], [9], 
#                [12], [13]]

which_files = [[12], [13]] # vacuum only


files = ["CO_299x70x15046.npy",  # 9
         "CO_199x70x15046.npy",  # 10  
         "CO_499x70x15046.npy",  # 16 
         "CO_499x70x15046.npy",  # 21 
         "CO_499x70x15046.npy",  # 25 
         
         "CO_249x70x15046.npy",  # 12 
         "CO_249x70x15046.npy",  # 13 
         
         "CO_499x70x15046.npy",  # 5 
         "CO_499x70x15046.npy",  # 9 
         "CO_499x70x15046.npy",  # 15 
         "CO_70x70x15046.npy",  # 17
         "CO_429x70x15046.npy", # 18
         
         "vacuum_background_132804x15046.bin", # vac on 6/21
         "vacuum_background_132804x15046.bin"] # vac on 6/23

folders = [r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_9\\",  
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_10\\", 
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_16\\", 
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_21\\",
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_25\\",
           
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-22-22\battalion_12\\",
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-22-22\battalion_13\\", 
           
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_5\\", 
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_9\\", 
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_15\\",
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_17\\",
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_18\\", 
           
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\Vaccum_Background\\", 
           r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\Vacuum_Background\\"]


for i, which in enumerate(which_files): 
  
    print(which_set[i])
    
    if len(which)==1:
        try: 
            IG_all = np.load(folders[which[0]] + files[which[0]])
        
        except: 
            IG_all = np.fromfile(folders[which[0]] + files[which[0]])
            shape = files[which[0]].split('_')[-1].split('.')[0].split('x')
            IG_all = IG_all.reshape(33201, int(shape[1]))
        
    elif len(which)==2: 
               
        IG_all = np.vstack((np.load(folders[which[0]] + files[which[0]]), np.load(folders[which[1]] + files[which[1]])))
    
    asdfsdsd
    
    IG_avg = np.mean(IG_all, axis=0)
    
    np.save(save_path+which_set[i], IG_avg)
    
    
    
    
    
    
    
    




