# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:02:06 2022

@author: scott
"""


import numpy as np




which_set = ['1ArHe', '2ArHe', '3ArHe', '4ArHe',
             '1ArL', # with bad lasers (?)
             '1Ar', '2Ar', '3Ar', '4Ar']

which_files = [[0,1], [2], [3], [4],
               [5,6],
               [10,11], [7], [8], [9]]


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
         "CO_429x70x15046.npy"]  # 18

folders = [r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_9\\", 
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_10\\", 
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_16\\", 
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_21\\",
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-21-22\PHASE_CORRECTED_DATA\battalion_25\\",
           
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-22-22\battalion_12\\",
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-22-22\battalion_13\\", 
           
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_5\\", 
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_9\\", 
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_15\\",
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_17\\",
           r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\06-23-22\PHASE_CORRECTED_DATA\battalion_18\\"]



for i, which in enumerate(which_files): 
    
    print(which_set[i])
    
    if len(which)==1:
        
        IG_all = np.load(folders[which[0]] + files[which[0]])
        
    elif len(which)==2: 
               
        IG_all = np.vstack((np.load(folders[which[0]] + files[which[0]]), np.load(folders[which[1]] + files[which[1]])))
    
    IG_avg = np.mean(IG_all, axis=0)
    
    np.save(which_set[i], IG_avg)
    
    
    
    
    
    
    
    




