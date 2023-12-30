r'''

silmaril 5 - ASC cutoff

prepares transmission data for labfit .ASC file format (includes provision for saturated features that are below a specified cutoff threshold)

r'''



import numpy as np
import pickle 
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

from scipy.constants import speed_of_light
import pldspectrapy as pld
import td_support as td

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

import time

# %% dataset specific information


save_data = True

folder_save = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - CO'

[fit_results_global, trans_all] = np.load('fit results for labfit.npy', allow_pickle=True)

meas_file_names = ['1Ar', '1ArL', '2Ar', '3Ar', '4Ar', '1ArHe', '2ArHe', '3ArHe', '4ArHe']

molecule_id = 5
PL = 1.27 # cm length of shok tube
y_CO = 0.05 # concentration in pure CO data

file_number_base = 1000

# %% iterate through the files and prepare the conditions for labfit   

for meas_file in meas_file_names: 
    
    fit_results_meas = fit_results_global[meas_file]
    
    trans_meas = trans_all[meas_file]
    wvn_full = trans_meas[0]
    
    shift = 0 # np.mean(fit_results_meas[:,6])
    
    print(meas_file)
    print('     {:.5f}    +/-     {:.5f}'.format(shift, np.std(fit_results_meas[:,6])))
    
    wvn_full+= shift
    
    file_number_base +=100
    file_number_iter = 0
    
    for i_file in range(len(fit_results_meas[:,0])): 
        
        
        trans = trans_meas[i_file+1]
        
        wvn = wvn_full[2000:]
        trans = trans[2000:]
        
        P = fit_results_meas[i_file,1] * 760 # convert from atm to Torr 
        T = fit_results_meas[i_file,2] - 273.15 # in C
        L = PL / 100 # in m

        t = int(fit_results_meas[i_file,0])

               
    # %% process and save ASC file for labfit   
    
        file_number_iter +=1 
        labfitname = str(file_number_base + file_number_iter).zfill(4)
                
        descriptor = 'Shock tube measurment of CO from June 2022 test {} at t = {}'.format(meas_file, t)
        
        '''
        Constructs Labfit input .asc file
        
        dataarray: input array of reduced comb data
        labfitname: 4 digit labfit identifier (str)
        descriptor: Description (str)
        T: temperature (C)
        P: pressure (torr)
        L: pathlength (m)
        yh2o: molefraction
        nuLow: low range of frequencies (cm-1), set to 0 for no filtering, otherwise use nuLow and nuHigh to define a subset of the spectrum in dataarray that is passed to the .asc file
        nuHigh: high range of frequencies
        mnum: molecule ID (hitran)
        
        '''
        
        np.set_printoptions(15)
                
        # format main values for labfit file
        Lstr = '%.7f' % L
        Tstr = ('%.4f' % T).rjust(12)
        Pstr = ('%.5f' % P).rjust(13)
        ystr = '%.6f' % y_CO
            
        delta = '%.30f' %  np.mean(np.diff(wvn)) #cm-1 ; note notation suppresses scientific notation
                
        trans = 100*trans # I had it scaled 0-1 (labfit wants 0-100)
                
        wvn_start = '%.10f' % wvn[0]
        wvn_stop = '%.10f' % wvn[-1]
        
        fname = labfitname + "_" + meas_file + "_" + str(t) + ".asc"
        
        if save_data: 
            
            file = open(os.path.join(folder_save,fname),'w')
            file.write("******* File "+labfitname+", "+descriptor+"\n")
            file.write(labfitname + "   " + wvn_start + "   " + wvn_stop + "   " + delta+"\n")
            file.write("  00000.00    0.00000     0.00e0     " + str(molecule_id) + "   2     3   0        0\n")
            file.write("    " + Lstr + Tstr + Pstr + "    " + ystr + "    .0000000 .0000000 0.000\n")
            
            file.write("    0.0000000     23.4486      0.00000    0.000000    .0000000 .0000000 0.000\n") # nothing
            
            file.write("    0.0000000     23.4486      0.00000    0.000000    .0000000 .0000000 0.000\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("DATE " + time.strftime("%m/%d/%Y") + "; time " + time.strftime("%H:%M:%S") + "\n")
            file.write("\n")
            # this line is an artifact of Ryan Cole. I'm not sure what the hard-coded numbers mean - scott
            file.write(wvn_start + " " + delta + " 15031 1 1    2  0.9935  0.000   0.000 294.300 295.400 295.300   7.000  22.000 500.000 START\n")
            
            wavelist = wvn.tolist()
            translist = trans.tolist()
            
            for i in range(len(wavelist)):
                wavelist[i] = '%.10f' %  wavelist[i]
                translist[i] = '%.10f' % translist[i]
                file.write(wavelist[i] + "      " + translist[i] + "\n")
            file.close()
            
            # print("Labfit Input Generated for Labfit file " + labfitname)
            # these values are also in the inp file (which I think labfit prefers to use) You will want to change them there to match the ASC file (if needed)
            # print(str(t) + '          ' + delta[:8] + '       ' + str(T + 273.15).split('.')[0] + '     ' + Pstr[:-4] + '         ' + ystr[:5] + '\n') 
        
    
