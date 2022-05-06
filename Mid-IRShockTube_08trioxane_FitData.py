#%% -------------------------------------- load some libraries -------------------------------------- 

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light
from scipy import signal

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td # time domain support
import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

# if you want to run a bunch of files at the same time, you can cue them with this
import time 
import psutil
while psutil.cpu_percent() > 80: time.sleep(60*15) # hang out for 15 minutes if CPU is busy

pld.db_begin('linelists')

#%% -------------------------------------- inputs we generally change -------------------------------------- 

num_IGs_avg = 3 # how many IGs to average together

f_counter_n = 9998061.1 # nominal near 10 MHz reading of the counter
ppIG = 17507
num_IGs = 50


#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 
      
forderLP = 2
fcutoffLP = 0.15

T_which = 0 # 0 (1400), 1 (1600), 2 (1800) 
H_precursor = False

baseline_TD_start = 20
baseline_TD_stop = 0 # 0 is none, high numbers start removing high noise datapoints (doesn't seem to change much)

T_fit_which = 'CO' # which molecule to use to fit temperature, True = all 

#%% -------------------------------------- generally applicable model conditions -------------------------------------- 

P = 5.0 # pressure in atm
PL = 1.27 # cm length of cell in furnace (double pass)

T_all = [1400, 1600, 1800]  # temperature in K
T = T_all[T_which]

#%% -------------------------------------- start dict that will hold data for the different molecules -------------------------------------- 

data = {}
molecules_meas = ['CO', 'H2CO']

for i, molecule in enumerate(molecules_meas): 

    data[molecule] = {}
    
    if molecule == 'CO': 
        data[molecule]['file_meas'] = 'co_surf_27_and_28_avged_across_shocks_50x17507.bin'
        data[molecule]['file_vac'] = 'co_vacuum_background.bin'
        
        data[molecule]['wvn2_fit'] = [10000/4.62, 10000/4.40]
        data[molecule]['wvn2_plot'] = [2187,2240]
        data[molecule]['molecule_ids'] = 5
        
        if H_precursor: data[molecule]['y_expected'] = [3E-5, 0.04, 0.08][T_which]
        else: data[molecule]['y_expected'] = [1E-7, 0.001, 0.075][T_which]
       
    elif molecule == 'H2CO': 
        data[molecule]['file_meas'] = 'h2co_surf_27_and_28_avged_across_shocks_50x17507.bin'
        data[molecule]['file_vac'] = 'h2co_vacuum_background.bin'
        
        data[molecule]['wvn2_fit'] = [10000/3.59, 10000/3.44]
        data[molecule]['wvn2_plot'] = [2787,2887]
        data[molecule]['molecule_ids'] = 20
        if H_precursor: data[molecule]['y_expected'] = [0.20, 0.16, 0.10][T_which]
        else: data[molecule]['y_expected'] = [0.20, 0.20, 0.12][T_which]
        
    if molecule == T_fit_which: data[molecule]['calc_T'] = True # calc T needs to be the first listed in molecules_meas so we do that first
    else: data[molecule]['calc_T'] = False
        
    data[molecule]['folder'] = r'H:\ShockTubeData\\'
    
    
    #%% -------------------------------------- load data -------------------------------------- 

    # read in the IGs (averaged between shocks, not averaged in time)
    data[molecule]['IG_all'] = np.fromfile(data[molecule]['folder']+data[molecule]['file_meas'])
    data[molecule]['IG_all'] = data[molecule]['IG_all'].reshape((num_IGs, ppIG))
    
    # load in the vacuum scan and smooth it
    data[molecule]['IG_vac'] = np.fromfile(data[molecule]['folder']+data[molecule]['file_vac'])
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

#%% -------------------------------------- average IGs together as desired (loop it) -------------------------------------- 

num_iters = num_IGs-num_IGs_avg # how many chunks num_IGs_avg wide can we make? 1+2+3, 2+3+4, 3+4+5, etc.
fit_results = np.zeros((num_iters+1,6*len(molecules_meas))) # number of iterations, 14 entries per iteration

for ig_start in range(num_iters+1): 
    
    ig_stop = ig_start + num_IGs_avg

    print('*************************************************')
    print('****** IG start:'+str(ig_start)+ ' IG stop:' + str(ig_stop)+' ******************')
    print('*************************************************')

    for i, molecule in enumerate(molecules_meas): 
        
        print('-----------------------------------------------')
        print('---------------------- fitting for:' + molecule + ' ----------')
        print('-----------------------------------------------')       
        
        # average IGs together
        IG_avg = np.mean(data[molecule]['IG_all'][ig_start:ig_stop,:],axis=0)
        
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
      
        mod, pars = td.spectra_single_lmfit() 
        
        pars['mol_id'].value = data[molecule]['molecule_ids']
        pars['shift'].vary = True
                                     
        pars['pathlength'].set(value = PL, vary = False)
        pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = False)
        pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = data[molecule]['calc_T'], min=250, max=3000)
        pars['molefraction'].set(value = data[molecule]['y_expected'], vary = True)
                    
        # TD_model_expected = mod.eval(xx=data[molecule]['wvn'], params=pars, name=molecule)

#%% -------------------------------------- trim to size and fit the spectrum -------------------------------------- 

        i_fits = td.bandwidth_select_td(data[molecule]['wvn'], data[molecule]['wvn2_fit'], max_prime_factor=50) # wavenumber indices of interest

        wvn_fit = data[molecule]['wvn'][i_fits[0]:i_fits[1]]
        trans_meas_fit = trans_meas[i_fits[0]:i_fits[1]]
               
        print('fitting spectra for ' + molecule)
        
        TD_meas_fit = np.fft.irfft(-np.log(trans_meas_fit))
        weight = td.weight_func(len(trans_meas_fit), baseline_TD_start, baseline_TD_stop) 
               
        if data[molecule]['calc_T'] is False and T_fit_which is not False: # if we aren't calcuating T this time, but we did already calculate it
            pars['temperature'].value = T_fit # use T_fit_which's temperature (should already be a fixed value)
            pars['shift'].value = shift_fit # something else to help things along for the weaker absorber
       
        fit = mod.fit(TD_meas_fit, xx = wvn_fit, params = pars, weights = weight, name=molecule)
        
        if molecule == T_fit_which: 
            T_fit = fit.params['temperature'].value # snag this temperature if we'll need it for the next molecule
            shift_fit = pars['shift'].value
        
        fit_results[ig_start, 6*i] = fit.params['temperature'].value
        fit_results[ig_start, 6*i+1] = fit.params['temperature'].stderr
        fit_results[ig_start, 6*i+2] = fit.params['shift'].value
        fit_results[ig_start, 6*i+3] = fit.params['shift'].stderr
        fit_results[ig_start, 6*i+4] = fit.params['molefraction'].value
        fit_results[ig_start, 6*i+5] = fit.params['molefraction'].stderr
               

    #%% save figure as you go so you can make a movie later (#KeepingUpWithPeter)
           
        TD_model_fit = fit.best_fit
        weight = fit.weights
        # plot frequency-domain fit
        trans_meas_noBL = np.real(np.fft.rfft(TD_meas_fit - (1-weight) * (TD_meas_fit - TD_model_fit)))
        trans_model = np.real(np.fft.rfft(TD_model_fit))
        # plot with residual
        fig, axs = plt.subplots(2,2, sharex = 'col', sharey = 'row', figsize=(10, 4),
                                gridspec_kw={'height_ratios': [3,1], 'width_ratios': [3,1], 'hspace':0.015, 'wspace':0.005})
        
        # title
        T_plot = str(int(np.round(fit.params['temperature'].value,0)))
        y_plot = str(np.round(fit.params['molefraction'].value*100,1))
        t_plot = str(int(np.round((ig_start + num_IGs_avg / 2)/dfrep*1e6,0)))
        P_plot = str(np.round(P,1))
        
        plt.suptitle('{} at {} K, {} atm, and {}% while averaging {} IG, ~{} us post shock'.format(molecule, T_plot, P_plot, y_plot, num_IGs_avg, t_plot))
        
        # fit plot to the right range
        i_range = td.bandwidth_select_td(wvn_fit, data[molecule]['wvn2_plot'])
        wvn_plot = wvn_fit[i_range[0]:i_range[1]]
        wvl_plot = 10000 / wvn_plot
        trans_meas_noBL = trans_meas_noBL[i_range[0]:i_range[1]]
        trans_model = trans_model[i_range[0]:i_range[1]]
        
        # top left plot - absorbance over wavelength for both model and meas
        axs[0,0].plot(wvl_plot, trans_meas_noBL, label='meas')
        axs[0,0].plot(wvl_plot, trans_model, label='model')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_ylabel('Absorbance')
        
        if molecule == 'CO': axs[0,0].set_ylim(-0.25, 1.5)
        elif molecule == 'H2CO': axs[0,0].set_ylim(-0.05, 0.15)

        # bottom left plot - absorbance over wavelength for both model and meas        
        axs[1,0].plot(wvl_plot, trans_meas_noBL - trans_model, label='meas-model')
        axs[1,0].legend(loc='upper right')
        axs[1,0].set_ylabel('Residual')
        axs[1,0].set_xlabel('Wavelength (um)')

        
        if molecule == 'CO': axs[1,0].set_ylim(-0.15, 0.15)
        elif molecule == 'H2CO': axs[1,0].set_ylim(-0.15, 0.15)
        
        
        # top right plot - absorbance over some wavelength for both model and meas
        axs[0,1].plot(wvl_plot, trans_meas_noBL, label='meas')
        axs[0,1].plot(wvl_plot, trans_model, label='model')
        
        if molecule == 'CO': axs[0,1].set_xlim(4.4809, 4.4868)
        elif molecule == 'H2CO': axs[0,1].set_xlim(3.539, 3.546)
           
        # bottom right plot - absorbance over some wavelength for both model and meas        
        axs[1,1].plot(wvl_plot, trans_meas_noBL - trans_model, label='meas-model')
        
        plt.savefig(os.path.abspath('')+r'/plots/{} {} averaging {}.jpg'.format(molecule, ig_start, num_IGs_avg), bbox_inches='tight')

        



