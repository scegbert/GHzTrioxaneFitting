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

#%% -------------------------------------- inputs we generally change -------------------------------------- 

f_counter_n = 9998061.1 # nominal near 10 MHz reading of the counter

fit_pressure = False # <--------------- use wisely, probably need to update CO for argon broadening first (also would want to link pressures first as well)
plot_fits = False
save_fits = True

T_fit_which = 'CO' # which molecule to use to fit temperature, True = all, False = None (fixed) 
# will now automatically put T_fit_which first in fitting order

# bins_avg = 16 # how many bins to average together (if #bins=8, bins_avg=3 means each iteration is ~3/8 * 1/dfrep)
# bins_step = 8 # how many bins to skip between averages

ig_start = 0 # start processing IG's at #ig_start
ig_stop = 49 # assume the process has completed itself by ig_stop 
ig_avg = 1 # how many to average together

ig_inc_shock = 19.5 # average location of the incident shock (this is what the data is clocked off of)
t_inc2ref_shock = 35 # time between incident and reflected shock

molecules_meas = ['CO', 'H2CO', 'C3H6O3']

fits_plot = ['temperature', 'molefraction','shift']
if fit_pressure: fits_plot.append('pressure') # if we're fitting it, you probably want to save it

#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 
      
forderLP = 2
fcutoffLP = 0.15 # low pass filter for smoothing the vacuum scan

baseline_TD_start = 20
baseline_TD_stop = 0 # 0 is none, high numbers start removing high noise datapoints (doesn't seem to change much)

H_precursor = False

#%% -------------------------------------- generally applicable model conditions -------------------------------------- 

T_pre = 300 # temperature in K before the shock (for scaling trioxane measurement)
P_pre = 0.1 # pressure in atm before the shock (for scaling trioxane measurement)
y_trioxane = 0.037/3 # guess at trioxane concentration (number doesn't matter, just for scaling)

P = 5.0 # pressure in atm after shock (currently assuming constant P)
PL = 1.27 # cm length of cell in furnace (double pass)

T_all = [1400, 1600, 1800]  # temperature in K
T_which = 0 # which of the T's above are we measuring? (feeds guesses for model of T and y)
T = T_all[T_which]

#%% -------------------------------------- start dict that will hold data for the different molecules -------------------------------------- 

data = {}
molecules_meas.insert(0, molecules_meas.pop(molecules_meas.index(T_fit_which))) # bring T_fit_which to the beginning of this list

for i, molecule in enumerate(molecules_meas): 

    data[molecule] = {}
    
    if molecule == 'CO': 
        data[molecule]['file_meas'] = ['batt_4_co_avged_shocks_20_before_50_after.npy', 'batt_5_co_avged_shocks_20_before_50_after.npy'] # should be list
        data[molecule]['file_vac'] = '4.5um_filter_phase_corrected.npy'
        
        
        data[molecule]['wvn2_fit'] = [10000/4.62, 10000/4.40]
        data[molecule]['wvn2_plot'] = [2187,2240]
        data[molecule]['molecule_ids'] = 5
        
        if H_precursor: data[molecule]['y_expected'] = [3E-5, 0.04, 0.08][T_which]
        else: data[molecule]['y_expected'] = [1E-7, 0.001, 0.075][T_which]
       
    elif molecule == 'H2CO': 
        data[molecule]['file_meas'] = ['batt_4_h2co_avged_shocks_20_before_50_after.npy', 'batt_5_h2co_avged_shocks_20_before_50_after.npy']
        data[molecule]['file_vac'] = '3.5um_filter_phase_corrected.npy'
        
        data[molecule]['wvn2_fit'] = [10000/3.59, 10000/3.44]
        data[molecule]['wvn2_plot'] = [2787,2887]
        data[molecule]['molecule_ids'] = 20
        if H_precursor: data[molecule]['y_expected'] = [0.20, 0.16, 0.10][T_which]
        else: data[molecule]['y_expected'] = [0.20, 0.20, 0.12][T_which]
        
    elif molecule == 'C3H6O3': 
        data[molecule]['file_meas'] = ['batt_4_h2co_avged_shocks_20_before_50_after.npy', 'batt_5_h2co_avged_shocks_20_before_50_after.npy'] # data['H2CO']['file_meas']
        data[molecule]['file_vac'] = '3.5um_filter_phase_corrected.npy' # data['H2CO']['file_vac']
        
        data[molecule]['wvn2_fit'] = [10000/3.52, 10000/3.50]
        data[molecule]['wvn2_plot'] = [2787,2887]
                
    if molecule == T_fit_which: data[molecule]['calc_T'] = True # calc T needs to be the first listed in molecules_meas so we do that first
    else: data[molecule]['calc_T'] = False
        
    data[molecule]['folder_data'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\trioxane\\'
    data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\trioxane\vacuum_bckgnd\\' # best for batt5_h2co
    
    # data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\vacuum_bckgnd_after_cleaning\\'
    # data[molecule]['folder_vac'] = r'H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\vacuum_bckgnd_end_of_experiment\\'

    
    #%% -------------------------------------- load data -------------------------------------- 
    
    # read in the IGs (averaged between shocks, not averaged in time or between batts) for all files in list, then average the lists together 
    # IG_all = np.mean(np.array([np.load(data[molecule]['folder_data'] + file) for file in data[molecule]['file_meas']]), axis=0) 
    IG_all = (np.load(data[molecule]['folder_data'] + data[molecule]['file_meas'][0]) + np.load(data[molecule]['folder_data'] + data[molecule]['file_meas'][1])) /2
    
    IG_shape = np.shape(IG_all)
    ppIG = IG_shape[-1]

    data[molecule]['IG_all'] = IG_all # .reshape((IG_shape[0] * IG_shape[1], IG_shape[2]), order='F') # this helped with time binning, no longer used

    print('*************************************************')
    print('****** {} IG shape is {}, ******************'.format(molecule, IG_shape))
    print('*************************************************')
    
    # load in the vacuum scan and smooth it
    data[molecule]['IG_vac'] = np.load(data[molecule]['folder_vac']+data[molecule]['file_vac'])
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
            pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = fit_pressure)
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
               
        if data[molecule]['calc_T'] is False and T_fit_which is not False: # if we aren't calcuating T this time, but we did already calculate it
            # this gets mad because it needs to fit T for T_fit and then it will come back and list that as the T for the second fit
            
            if y_min > 0.002: 

                pars['temperature'].value = T_fit # use T_fit_which's temperature (should already be a fixed value)
                pars['shift'].value = shift_fit # something else to help things along for the weaker absorber

            else: 

                pars['temperature'].set(value = T_pre, vary=False) # we couldn't see T_fit_which, don't use it's temperature
                pars['pressure'].set(value = P_pre) # we couldn't see T_fit_which, don't use it's temperature

        if molecule == 'C3H6O3':
            
            pars['shift'].set(value=0, vary=False)
            fit = mod.fit(TD_meas_fit, xx = wvn_fit, xx_HITRAN=wvn_fit, coef_HITRAN=data[molecule]['trioxane_ref_coefs'], 
                          weights = weight, params = pars)
            
        else: 
            
            fit = mod.fit(TD_meas_fit, xx = wvn_fit, params = pars, weights = weight, name=molecule)
        
        if molecule == T_fit_which: 
            T_fit = fit.params['temperature'].value # snag this temperature if we'll need it for the next molecule
            y_fit = fit.params['molefraction'].value # snag this concentration to see if we trust the temperature
            
            try: y_min = fit.params['molefraction'].value - fit.params['molefraction'].stderr # minimum predicted concentration for t_fit_which (is it even there)
            except: y_min = fit.params['molefraction'].value - 0.002
            
            shift_fit = pars['shift'].value
        
        for i_results, which_results in enumerate(fits_plot): 
            
            # save some fit results and errors for plotting later
            fit_results[i_ig, 6*i_molecule+2*i_results+1] = fit.params[which_results].value
            fit_results[i_ig, 6*i_molecule+2*i_results+2] = fit.params[which_results].stderr
               
#%% --------------------------------------  save figure as you go so you can make a movie later (#KeepingUpWithPeter) -------------------------------------- 
        ig_avg_location = (ig_start_iter + ig_stop_iter-1)/2 - ig_inc_shock # average full IG periodes post shock\
        fit_results[i_ig, 0] = ig_avg_location/dfrep*1e6 - t_inc2ref_shock # time referenced to the reflected shock wave
        t_plot = str(int(np.round(fit_results[i_ig, 0])))
                
        if plot_fits: 
        
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
            P_plot = str(np.round(P,1))
                    
            plt.suptitle('{} at {} K, {} atm, and {}% while averaging {} bins, ~{} us post shock'.format(molecule, T_plot, P_plot, y_plot, ig_avg, t_plot))
            
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
            
            if save_fits: 
                
                plt.savefig(os.path.abspath('')+r'\plots\fitting {} using T_{} while averaging {} IGs starting with IG {}.jpg'.format(
                    molecule, T_fit_which, ig_avg, ig_start_iter), bbox_inches='tight')
            
            plt.close()

        if save_fits: 
            
            np.save(os.path.abspath('')+r'\plots\fit results using T_{} while averaging {} IGs'.format(T_fit_which, ig_avg), fit_results)
        
#%% -------------------------------------- plot the fit results with error bars -------------------------------------- 

fit_results = np.load(os.path.abspath('')+r'\plots\fit results using T_{} while averaging {} IGs.npy'.format(T_fit_which, ig_avg))

# fit_results = fit_results[fit_results[0,:] != 0,:]

x_offset = 0.15

for i_results, which_results in enumerate(fits_plot): 

    plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    plt.title('T_{} while averaging {} IGs.npy'.format(T_fit_which, ig_avg)) 

    for i_molecule, molecule in enumerate(molecules_meas): 
    
        # for bin_avg in [1,3,6]: 
            
        #     if bin_avg == 1: fit_results = fit_results1CO
        #     elif bin_avg == 3: fit_results = fit_results3CO
        #     elif bin_avg ==6: fit_results = fit_results6CO
            
        plot_x = fit_results[:,0] + x_offset*i_molecule
        
        plot_y = fit_results[:, 6*i_molecule+2*i_results+1]
        plot_y_unc = fit_results[:, 6*i_molecule+2*i_results+2]
        
        if molecule == 'C3H6O3' and i_results == 1: 
            plot_y = plot_y/9
            plot_y_unc = plot_y_unc/9
             
        plt.errorbar(plot_x, plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
        plt.plot(plot_x, plot_y, marker='x', label='{} over {} IGs'.format(molecule, ig_avg) , zorder=2)
        
        
    plt.xlabel('Time Post Shock (us)')
    plt.ylabel('{}'.format(which_results))
    
    plt.legend()
    
    
        
