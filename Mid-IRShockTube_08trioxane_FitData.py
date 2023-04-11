#%% -------------------------------------- load some libraries -------------------------------------- 

# delay until the processor is running below XX% load
import time 
import psutil
# while psutil.cpu_percent() > 80: time.sleep(60*15) # hang out for 15 minutes if CPU is busy

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light
from scipy import signal
from scipy import interpolate

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td # time domain support
import hapi as hapi

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

pld.db_begin('linelists')

#%% -------------------------------------- inputs we generally change -------------------------------------- 

f_counter_n = 9998061.1 # nominal near 10 MHz reading of the counter

ig_start = 0 # start processing IG's at #ig_start
ig_stop = 500 # assume the process has completed itself by ig_stop 
ig_avg = 10 # how many to average together

ig_inc_shock = 130 # average location of the incident shock (this is what the data is clocked off of)
t_inc2ref_shock = 0 # time between incident and reflected shock in microseconds

molecules_meas = ['C3H6O3'] # ['CO','C3H6O3']

fits_plot = ['temperature', 'pressure', 'molefraction','shift']

#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 

forderLP = 2
fcutoffLP = 0.15 # low pass filter for smoothing the vacuum scan

baseline_TD_start = 20
baseline_TD_stop = 0 # 0 is none, high numbers start removing high noise datapoints (doesn't seem to change much)

#%% -------------------------------------- generally applicable model conditions -------------------------------------- 

T_pre = 300 # temperature in K before the shock (for scaling trioxane measurement)
P_pre = 0.44 # pressure in atm before the shock (for scaling trioxane measurement)
y_trioxane = 0.026 # calculated trioxane concentration

PL = 1.27 # cm length of cell in furnace (double pass)

# pressure and time values from Nazanin's spreadsheet for surfs 27+28 (april data)
P_t_ = [3.0000, 3.0000, 3.0010, 4.7705, 5.0586, 5.5368, 6.0434, 6.3310, 6.4221, 6.8459, 7.0457, 7.1973, 7.3007, 7.5591,
       7.8027, 7.8948, 7.7921, 7.9851, 7.9902, 7.9014, 7.6677, 7.6831, 7.7936, 7.6001, 7.3926, 7.4636, 7.1844, 6.9945, 
       6.5092, 6.3817, 6.2883, 6.1634, 5.9154, 6.0739, 5.9183, 5.8858]
t_P = [-35.0, -17.5, 0.0, 17.5, 35.0, 52.5, 70.0, 87.5, 105.0, 122.5, 140.0, 157.5, 175.0, 192.5, 210.0, 227.5, 
       245.0, 262.5, 280.0, 297.5, 315.0, 332.5, 350.0, 367.5, 385.0, 402.5, 420.0, 437.5, 455.0, 472.5, 490.0, 
       507.5, 525.0, 542.5, 560.0, 577.5]

T_t = [280.57, 296.21, 747.00, 1087.70, 1230.10, 1370.53, 1517.92, 1525.57, 1498.45, 1504.34, 1546.08 ,1556.38, 1558.67,
       1585.41, 1554.51, 1547.73 ,1561.39, 1574.17, 1567.93, 1541.41, 1545.18, 1560.10, 1556.79, 1532.64, 1528.45, 1515.90,
       1450.16, 1448.82, 1445.73, 1418.62, 1403.51, 1399.92, 1417.25, 1366.89, 1337.77, 1340.51, 1335.29, 1354.32, 1359.77, 
       1350.70, 1354.20, 1350.48, 1333.43, 1345.09, 1354.16, 1369.27, 1366.75, 1385.33, 1395.50, 1397.05]

t_T = [-35, -17.494, 0.0119324, 17.5179, 35.0239, 52.5298, 70.0358, 87.5418, 105.048, 122.554, 140.06, 157.566, 175.072, 
       192.578, 210.084, 227.589, 245.095, 262.601, 280.107, 297.613 ,315.119 ,332.625, 350.131, 367.637, 385.143 ,402.649,
       420.155, 437.661, 455.167, 472.673, 490.179, 507.685, 525.191, 542.697, 560.203, 577.709, 595.215, 612.721, 630.227, 
       647.733, 665.239, 682.745, 700.251, 717.757, 735.263, 752.768, 770.274, 787.78, 805.286, 822.792]

T_t = T_t[:35]
t_PTdata = t_T[:35]

P_t = interpolate.interp1d(t_P, P_t_)(t_PTdata)


#%% -------------------------------------- start dict that will hold data for the different molecules -------------------------------------- 

data = {}

for i, molecule in enumerate(molecules_meas): 
     
    data[molecule] = {}
    
    if molecule == 'CO': 
        # data[molecule]['file_meas'] = ['co_surf_27_and_28_avged_across_shocks_50x17507.bin'] # should be list
        # data[molecule]['file_vac'] = 'co_vacuum_background.bin'
    
        data[molecule]['file_meas'] = ['co_averaged_surfs_27_and_28.npy']  # should be list
        data[molecule]['file_vac'] = 'co_vacuum_bckgnd_avg.npy'
    
        data[molecule]['wvn2_fit'] = [10000/4.62, 10000/4.40]
        data[molecule]['wvn2_plot'] = [2187,2240]
        data[molecule]['molecule_ids'] = 5
        
        data[molecule]['y_expected'] = .07 # improved guess based on what we see [1E-7, 0.001, 0.075][T_which]
        
        data[molecule]['calc_T'] = True # calc T needs to be the first listed in molecules_meas so we do that first
        
    elif molecule == 'C3H6O3': 
        data[molecule]['file_meas'] = ['h2co_averaged_surfs_27_and_28.npy'] # data['H2CO']['file_meas']
        data[molecule]['file_vac'] = 'h2co_vacuum_bckgnd_avg.npy' # data['H2CO']['file_vac']
    
        data[molecule]['wvn2_fit'] = [10000/3.52, 10000/3.50]
        data[molecule]['wvn2_plot'] = [2787,2887]
                
        data[molecule]['calc_T'] = False
        
    data[molecule]['folder_data'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_1\\'
    data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_1\\' # best for batt5_h2co
    
    # data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\vacuum_bckgnd_after_cleaning\\'
    # data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\vacuum_bckgnd_end_of_experiment\\'
    
    
    #%% -------------------------------------- load data -------------------------------------- 
    
    # read in the IGs (averaged between shocks, not averaged in time or between batts) for all files in list, then average the lists together 
    IG_all = np.mean(np.array([np.load(data[molecule]['folder_data'] + file) for file in data[molecule]['file_meas']]), axis=0) # npy
    # IG_all = np.mean(np.array([np.fromfile(data[molecule]['folder_data'] + file) for file in data[molecule]['file_meas']]), axis=0) # bin 
    # IG_all = IG_all.reshape((50,17507)) # bin 
    
    IG_shape = np.shape(IG_all)
    ppIG = IG_shape[-1]
    
    data[molecule]['IG_all'] = IG_all 
    
    print('*************************************************')
    print('****** {} IG shape is {}, ******************'.format(molecule, IG_shape))
    print('*************************************************')
    
    # load in the vacuum scan and smooth it
    data[molecule]['IG_vac'] = np.load(data[molecule]['folder_vac']+data[molecule]['file_vac']) # npy
    # data[molecule]['IG_vac'] = np.fromfile(data[molecule]['folder_vac']+data[molecule]['file_vac']) # bin
    
    i_center = int((len(data[molecule]['IG_vac'])-1)/2)
    data[molecule]['meas_vac'] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data[molecule]['IG_vac']))).__abs__()[:i_center+1] # fft and remove reflected portion
    
    b, a = signal.butter(forderLP, fcutoffLP)
    data[molecule]['meas_vac_smooth'] = signal.filtfilt(b, a, data[molecule]['meas_vac'])
                
    #%% -------------------------------------- setup wavenumber axis based on lock conditions -------------------------------------- 
    
    wvn_target = np.mean(data[molecule]['wvn2_fit']) # a wavenumber we know is in our range
    hz2cm = 1 / speed_of_light / 100
    
    # sort out some frequencies
    frep_n = 1010e6 - f_counter_n
    dfrep = frep_n / ppIG 
    favg = frep_n-dfrep/2
    
    # calculate wavenumber (and wavelength) axis
    nyq_span = frep_n**2 / 2 / dfrep
    nyq_num = np.floor((wvn_target/hz2cm) / nyq_span).astype(int)
    
    nyq_start = nyq_num * nyq_span
    nyq_stop = nyq_start + nyq_span
    data[molecule]['wvn'] = np.arange(nyq_start, nyq_stop, favg) * hz2cm # convert from Hz to cm-1
    data[molecule]['wvl'] = 10000 / data[molecule]['wvn']    
    
    
    #%% -------------------------------------- setup trioxane reference measurement -------------------------------------- 
    
    if molecule == 'C3H6O3': 
        
        # read in the IGs (averaged between shocks, not averaged in time)
        IG_avg = np.mean(data[molecule]['IG_all'][0:int(ig_inc_shock)-1], axis=0) # use pre-shock IGs for trioxane baseline
        trans_meas = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_avg))).__abs__()[:i_center+1] / data[molecule]['meas_vac_smooth']
        
        i_fits = td.bandwidth_select_td(data[molecule]['wvn'], data[molecule]['wvn2_fit'], max_prime_factor=50) # wavenumber indices of interest
        trans_meas_trim = trans_meas[i_fits[0]:i_fits[1]]
    
        b, a = signal.butter(forderLP, fcutoffLP)
        trans_meas_trim_filt = signal.filtfilt(b, a, trans_meas_trim)
        abs_meas_trim_filt = -np.log(trans_meas_trim_filt)
        
        data[molecule]['trioxane_ref_coefs'] = abs_meas_trim_filt / PL / hapi.volumeConcentration(y_trioxane * P_pre, T_pre) # normalize to mimick HITRAN cross section
        
        
#%% -------------------------------------- average IGs together as desired (loop it) -------------------------------------- 

# program will loop through them like this (assuming bins_avg = 3, ig_start = 15): 15b1+15b2+15b3, 15b2+15b3+15b4, ..., 15b7+15b8+16b1, 15b8+16b1+16b3, ...
ig_start_iters = np.arange(ig_start, ig_stop - ig_avg+2)

fit_results = np.zeros((len(ig_start_iters),1+2*len(fits_plot)*len(molecules_meas))) 
# data format: [time, fits_plot[0]_molecule_1, fits_plot[0]_unc_molecule_1, fits_plot[1]_molecule_1, ..., fits_plot[0]_molecule_2, fits_plot[0]_unc_molecule_2, ...]

for i_ig, ig_start_iter in enumerate(ig_start_iters): 
    
    ig_stop_iter = ig_start_iter + ig_avg

    print('*************************************************')
    print('****** IG start:'+str(ig_start_iter)+ ' IG stop:' + str(ig_stop_iter-1)+' ******************')
    print('*************************************************')

    ig_avg_location = (ig_start_iter + ig_stop_iter - 1) / 2 - ig_inc_shock  # average full IG periodes post shock
    t_processing = ig_avg_location / dfrep * 1e6 - t_inc2ref_shock  # time referenced to the reflected Shock
    fit_results[i_ig, 0] = t_processing
    
    
    if t_processing < 0: 
        P = P_pre
        T = 300
    else: 
        P = P_t[np.argmin(abs(t_PTdata-t_processing))] 
        T = T_t[np.argmin(abs(t_PTdata-t_processing))] 

    for i_molecule, molecule in enumerate(molecules_meas): 
        
        # average IGs together
        IG_avg = np.mean(data[molecule]['IG_all'][ig_start_iter:ig_stop_iter,:],axis=0)
        
        meas_avg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_avg))).__abs__()
        i_center = int((len(meas_avg)-1)/2)
        meas_avg = meas_avg[:i_center+1] # remove reflected portion
        
        # divide by vacuum to mostly normalize things
        trans_meas = meas_avg / data[molecule]['meas_vac_smooth']
        
        # normalize max value to 1 (ish)
        wvn_target = np.mean(data[molecule]['wvn2_fit']) # a wavenumber we know is in our range
        i_target = np.argmin(abs(data[molecule]['wvn']-wvn_target))
        trans_meas = trans_meas / max(trans_meas[i_target-500:i_target+500])
    
#%% -------------------------------------- setup the model - re-initialize every time -------------------------------------- 
        if molecule == 'C3H6O3': 
    
            mod, pars = td.spectra_cross_section_lmfit() 
            
            pars['shift'].vary = True
                                         
            pars['pathlength'].set(value = PL, vary = False)
            pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = False)
            pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = data[molecule]['calc_T'], min=250, max=3000)
            pars['molefraction'].set(value = 0.001, vary = True) # starting out by assuming concentration is 0, TODO - make a better guess (?)

        else: 
           
            mod, pars = td.spectra_single_lmfit() 
            
            pars['mol_id'].value = data[molecule]['molecule_ids'] 
            pars['shift'].vary = True
                                         
            pars['pathlength'].set(value = PL, vary = False)
            pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = False, max = P*2, min = P/2)
            pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = data[molecule]['calc_T'], min=250, max=3000)
            pars['molefraction'].set(value = data[molecule]['y_expected']*np.random.rand()/1000, vary = True)
            
        # TD_model_expected = mod.eval(xx=data[molecule]['wvn'], params=pars, name=molecule)

#%% -------------------------------------- trim to size and fit the spectrum -------------------------------------- 

        i_fits = td.bandwidth_select_td(data[molecule]['wvn'], data[molecule]['wvn2_fit'], max_prime_factor=50) # wavenumber indices of interest

        wvn_fit = data[molecule]['wvn'][i_fits[0]:i_fits[1]]
        trans_meas_fit = trans_meas[i_fits[0]:i_fits[1]]
        
        print('fitting spectra for ' + molecule)
        
        TD_meas_fit = np.fft.irfft(-np.log(trans_meas_fit))
        weight = td.weight_func(len(trans_meas_fit), baseline_TD_start, baseline_TD_stop)         
        
        if molecule == 'C3H6O3':
            
            pars['shift'].set(value=0, vary=False)
            fit = mod.fit(TD_meas_fit, xx = wvn_fit, xx_HITRAN=wvn_fit, coef_HITRAN=data[molecule]['trioxane_ref_coefs'], 
                          weights = weight, params = pars)
            
        else: 
            
            fit = mod.fit(TD_meas_fit, xx = wvn_fit, params = pars, weights = weight, name=molecule)
        
        for i_results, which_results in enumerate(fits_plot): 
            
            # save some fit results and errors for plotting later
            fit_results[i_ig, 6*i_molecule+2*i_results+1] = fit.params[which_results].value
            fit_results[i_ig, 6*i_molecule+2*i_results+2] = fit.params[which_results].stderr
               
        
# rescaled the data so that the average in the 400-100 pre shock seconds was the correct average