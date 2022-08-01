#%% -------------------------------------- load some libraries -------------------------------------- 

# delay until the processor is running below XX% load
import time 
import psutil
while psutil.cpu_percent() > 80: time.sleep(60*15) # hang out for 15 minutes if CPU is busy

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light
from scipy import signal

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td # time domain support
import hapi as hapi

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

pld.db_begin('linelists')

#%% -------------------------------------- inputs we change sometimes -------------------------------------- 

f_counter_n = 10007604.8 # nominal near 10 MHz reading of the counter

fit_pressure = False # <--------------- use wisely, probably need to update CO for argon broadening first 
time_resolved_pressure = True

fit_concentration = False

data_folder = r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\\"

plot_fits = True
save_fits = True

ig_start = 0 # start processing IG's at #ig_start
ig_stop = 69 # assume the process has completed itself by ig_stop 
ig_avg = 1 # how many to average together

ig_inc_shock = 19.5 # average location of the incident shock (this is what the data is clocked off of)
t_inc2ref_shock = 35 # time between incident and reflected shock in microseconds

fits_plot = ['temperature', 'pressure', 'molefraction', 'shift']

#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 

forderLP = 2
fcutoffLP = 0.15 # low pass filter for smoothing the vacuum scan

baseline_TD_start = 20
baseline_TD_stop = 0 # 0 is none, high numbers start removing high noise datapoints (doesn't seem to change much)

#%% -------------------------------------- load and smooth the vacuum scan -------------------------------------- 

# IG_vac = np.load(data_folder + 'vac.npy')

# load in the vacuum scan and smooth it
# data[molecule]['IG_vac'] = np.load(data[molecule]['folder_vac']+data[molecule]['file_vac'])
# data[molecule]['IG_vac'] = np.fromfile(data[molecule]['folder_vac']+data[molecule]['file_vac'])

# i_center = int((len(data[molecule]['IG_vac'])-1)/2)
# data[molecule]['meas_vac'] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data[molecule]['IG_vac']))).__abs__()[:i_center+1] # fft and remove reflected portion

# b, a = signal.butter(forderLP, fcutoffLP)
# data[molecule]['meas_vac_smooth'] = signal.filtfilt(b, a, data[molecule]['meas_vac'])

meas_vac_smooth = np.ones(7523)
ppIG = 15046

#%% -------------------------------------- setup wavenumber axis based on lock conditions -------------------------------------- 

wvn_target = 2100 # a wavenumber we know is in our range
wvn2_fit = [2095, 2235]

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
wvn = np.arange(nyq_start, nyq_stop, favg)[:-1] * hz2cm # convert from Hz to cm-1
wvl = 10000 / wvn


#%% -------------------------------------- generally applicable model conditions -------------------------------------- 

molecule_name = 'CO'
molecule_id = 5
PL = 1.27 # cm length of cell in furnace (double pass)
y_CO = 0.05

T_pre = 300 # temperature in K before the shock (for scaling trioxane measurement)
P_pre = 0.44 # pressure in atm before the shock (for scaling trioxane measurement)

# P_all =           [    3,      3,     5,     3,     3,      3,        5,       3,       3] # pressure in atm after shock (if assuming constant P)
T_all =           [ 1200,   1200,  1200,  1500,  1820,  1200,      1200,    1500,    1820]  # temperature in K
meas_file_names = ['1Ar', '1ArL', '2Ar', '3Ar', '4Ar', '1ArHe', '2ArHe', '3ArHe', '4ArHe']


#%% -------------------------------------- setup for given file and load measurement data -------------------------------------- 

fit_results = {}

for i_file, meas_file in enumerate(meas_file_names): 

    T = T_all[i_file]
    # P = P_all[i_file]
    
    # load time resolved pressure data
    pressure_data = np.loadtxt(r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\Averaged Pressure Profile {}.csv".format(meas_file), delimiter=',')
    pressure_data_P = pressure_data[:,1] / 1.013 + 0.829 # convert from bar_gauge to ATM_abs
    pressure_data_t = pressure_data[:,0] * 1000 # convert to milliseconds
    
    pressure_data_P_smooth = pressure_data_P.copy() # smooth out the ringing in the pressure sensor
    b, a = signal.butter(forderLP, fcutoffLP)
    pressure_data_P_smooth[np.argmin(abs(pressure_data_t))+1:] = signal.filtfilt(b, a, pressure_data_P[np.argmin(abs(pressure_data_t))+1:])
        
    IG_all = np.load(data_folder+meas_file+'.npy') 
              
    #%% -------------------------------------- average IGs together as desired (loop it) -------------------------------------- 
    
    # program will loop through them like this (assuming bins_avg = 3, ig_start = 15): 15b1+15b2+15b3, 15b2+15b3+15b4, ..., 15b7+15b8+16b1, 15b8+16b1+16b3, ...
    ig_start_iters = np.arange(ig_start, ig_stop - ig_avg+2)
    
    fit_results[meas_file] = np.zeros((len(ig_start_iters),1+2*len(fits_plot))) 
    
    for i_ig, ig_start_iter in enumerate(ig_start_iters): 
        
        ig_stop_iter = ig_start_iter + ig_avg
    
        print('*************************************************')
        print('****** IG start:'+str(ig_start_iter)+ ' IG stop:' + str(ig_stop_iter-1)+' ******************')
        print('*************************************************')
    
        ig_avg_location = (ig_start_iter + ig_stop_iter - 1) / 2 - ig_inc_shock  # average full IG periodes post shock
        t_processing = ig_avg_location / dfrep * 1e6 - t_inc2ref_shock  # time referenced to the reflected Shock
        fit_results[meas_file][i_ig, 0] = t_processing
        
        if time_resolved_pressure: # if we want time resolved pressure
            P = pressure_data_P_smooth[np.argmin(abs(pressure_data_t-t_processing))] 
            
        # average IGs together
        IG_avg = np.mean(IG_all[ig_start_iter:ig_stop_iter,:],axis=0)
        
        meas_avg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_avg))).__abs__()
        i_center = int((len(meas_avg)-1)/2)
        meas_avg = meas_avg[:i_center+1] # remove reflected portion
        
        # divide by vacuum to mostly normalize things
        trans_meas = meas_avg / meas_vac_smooth
        
        # normalize max value to 1 (ish)
        i_target = np.argmin(abs(wvn-wvn_target))
        trans_meas = trans_meas / max(trans_meas[i_target-500:i_target+500])

        trans_meas = trans_meas[::-1] #<------ flip to agree with nyquist window 
        
#%% -------------------------------------- setup the model - re-initialize every time -------------------------------------- 
       
        mod, pars = td.spectra_single_lmfit() 
        
        pars['mol_id'].value = molecule_id
        pars['shift'].vary = True
                                     
        pars['pathlength'].set(value = PL, vary = False)
        pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = fit_pressure)
        
        if t_processing < 0: 
            pars['temperature'].set(value = T_pre + T*np.random.rand()/1000, vary = True, min=250, max=3000)
        else:
            pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = True, min=250, max=3000)
        
        pars['molefraction'].set(value = y_CO + y_CO*np.random.rand()/1000, vary = fit_concentration)
        
#%% -------------------------------------- trim to size and fit the spectrum -------------------------------------- 

        i_fits = td.bandwidth_select_td(wvn, wvn2_fit, max_prime_factor=50) # wavenumber indices of interest

        wvn_fit = wvn[i_fits[0]:i_fits[1]]
        trans_meas_fit = trans_meas[i_fits[0]:i_fits[1]]
                       
        TD_meas_fit = np.fft.irfft(-np.log(trans_meas_fit))
        weight = td.weight_func(len(trans_meas_fit), baseline_TD_start, baseline_TD_stop) 
        
        TD_model_expected = mod.eval(xx=wvn_fit, params=pars, name=molecule_name)
        abs_model_expected = np.real(np.fft.rfft(TD_model_expected))
        
        fit = mod.fit(TD_meas_fit, xx = wvn_fit, params = pars, weights = weight, name=molecule_name)
        
        for i_results, which_results in enumerate(fits_plot): 
            
            # save some fit results and errors for plotting later
            fit_results[meas_file][i_ig, 2*i_results+1] = fit.params[which_results].value
            fit_results[meas_file][i_ig, 2*i_results+2] = fit.params[which_results].stderr
               
#%% --------------------------------------  save figure as you go so you can make a movie later (#KeepingUpWithPeter) -------------------------------------- 

        if plot_fits: 
        
            TD_model_fit = fit.best_fit
            weight = fit.weights
            
            # plot frequency-domain fit
            abs_meas_noBL = np.real(np.fft.rfft(TD_meas_fit - (1-weight) * (TD_meas_fit - TD_model_fit)))
            abs_model = np.real(np.fft.rfft(TD_model_fit))
            wvl_plot = 10000 / wvn_fit
            
            # plot with residual
            fig, axs = plt.subplots(2,4, sharex = 'col', sharey = 'row', figsize=(12, 4),
                                    gridspec_kw={'height_ratios': [3,1], 'width_ratios': [5,1,1,1], 'hspace':0.015, 'wspace':0.005})
            
            # title
            t_plot = str(int(np.round(t_processing)))
            T_plot = str(int(np.round(fit.params['temperature'].value,0)))
            y_plot = str(np.round(fit.params['molefraction'].value*100,1)) 
            P_plot = str(np.round(fit.params['pressure'].value,1))
            
            plot_title = '{} at {} K and {} atm while averaging {} IGs ~{} us post shock'.format(meas_file, T_plot, P_plot, y_plot, ig_avg, t_plot)
            
            plt.suptitle(plot_title)
            
            # top first plot - absorbance over wavelength for both model and meas
            axs[0,0].plot(wvl_plot, abs_meas_noBL, label='meas')
            axs[0,0].plot(wvl_plot, abs_model, label='model')
            axs[0,0].legend(loc='upper right')
            axs[0,0].set_ylabel('Absorbance')
            
            axs[0,0].set_ylim(-0.25, 1.5)
    
            # bottom first plot - absorbance over wavelength for both model and meas        
            axs[1,0].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
            axs[1,0].legend(loc='upper right')
            axs[1,0].set_ylabel('Residual')
            axs[1,0].set_xlabel('Wavelength (um)')
            
            axs[1,0].set_ylim(-0.4, 0.4)
                    
            
            
            # top second plot - absorbance over some wavelength for both model and meas
            axs[0,1].plot(wvl_plot, abs_meas_noBL, label='meas')
            axs[0,1].plot(wvl_plot, abs_model, label='model')
            axs[0,1].set_xlim(4.4809, 4.4868)
               
            # bottom second plot - absorbance over some wavelength for both model and meas        
            axs[1,1].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
            
            
            
            
            
            # top third plot - absorbance over some wavelength for both model and meas
            axs[0,2].plot(wvl_plot, abs_meas_noBL, label='meas')
            axs[0,2].plot(wvl_plot, abs_model, label='model')
            axs[0,2].set_xlim(4.4809, 4.4868)
               
            # bottom third plot - absorbance over some wavelength for both model and meas        
            axs[1,2].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
            
            
            
            
            # top fourth plot - absorbance over some wavelength for both model and meas
            axs[0,3].plot(wvl_plot, abs_meas_noBL, label='meas')
            axs[0,3].plot(wvl_plot, abs_model, label='model')
            axs[0,3].set_xlim(4.4809, 4.4868)
               
            # bottom fourth plot - absorbance over some wavelength for both model and meas        
            axs[1,3].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
            
            
            asdfasdfsd
            
            if save_fits: 
                
                plt.savefig(os.path.abspath('')+r'\plots\{}.png'.format(plot_title), bbox_inches='tight')
            
            # plt.close()

        # if save_fits: 
            
        #     np.save(os.path.abspath('')+r'\plots\fit results using T_{} while averaging {} IGs'.format(T_fit_which, ig_avg), fit_results)
                
#%% -------------------------------------- plot the fit results with error bars -------------------------------------- 

# fit_results = np.load(os.path.abspath('')+r'\plots\fit results using T_{} while averaging {} IGs.npy'.format(T_fit_which, ig_avg))

# fit_results = fit_results[fit_results[0,:] != 0,:]

x_offset = 0.15

name = ['P = 5', 'P = 6', 'P(t)', 'P_optical']
i_molecule = 0


for i_results, which_results in enumerate(fits_plot): 

    plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    plt.title('T_{} while averaging {} IGs.npy'.format(T_fit_which, ig_avg)) 

    for i_fit, fit_results in enumerate([fit_results5, fit_results6, fit_results_t, fit_results_f]): 
    # for i_molecule, molecule in enumerate(molecules_meas): 
    
        # for bin_avg in [1,3,6]: 
            
        #     if bin_avg == 1: fit_results = fit_results1CO
        #     elif bin_avg == 3: fit_results = fit_results3CO
        #     elif bin_avg ==6: fit_results = fit_results6CO
            
        plot_x = fit_results[:,0] + x_offset*i_molecule
        
        if i_fit == 0 and which_results == 'pressure': 
            plot_y = np.ones_like(plot_x) * 5.0
            plot_y_unc = np.zeros_like(plot_x)
        
        elif i_fit == 1 and which_results == 'pressure': 
            plot_y = np.ones_like(plot_x) * 6.0
            plot_y_unc = np.zeros_like(plot_x)
        
        else: 
            plot_y = fit_results[:, 6*i_molecule+2*i_results+1]
            plot_y_unc = fit_results[:, 6*i_molecule+2*i_results+2]
        
        if molecule == 'C3H6O3' and i_results == 1: 
            plot_y = plot_y/9
            plot_y_unc = plot_y_unc/9
             
        plt.errorbar(plot_x, plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
        # plt.plot(plot_x, plot_y, marker='x', label='{} over {} IGs'.format(molecule, ig_avg) , zorder=2)
        plt.plot(plot_x, plot_y, marker='x', label=name[i_fit] , zorder=2)
        
        
    plt.xlabel('Time Post Shock (us)')
    plt.ylabel('{}'.format(which_results))
    
    plt.legend(loc='lower right')
    
        
    
plt.figure()    
plt.plot(TD_meas_fit)
plt.plot(TD_model_expected)

    
    
plt.figure()    
plt.plot(wvl_plot, abs_meas_noBL, label='meas', linewidth=5)
plt.plot(wvl_plot, abs_model, label='model')
plt.plot(wvl_plot, abs_model_expected, label='predicted')


        
