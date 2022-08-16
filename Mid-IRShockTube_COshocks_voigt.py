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
import linelist_conversions as db

import labfithelp as lab

from lmfit import Model

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()


#%% -------------------------------------- inputs we change sometimes -------------------------------------- 

f_counter_n = 10007604.8 # nominal near 10 MHz reading of the counter

fit_pressure = True # <--------------- use wisely, probably need to update CO for argon broadening first 
time_resolved_pressure = True
co_argon_database = True

fit_concentration = True
fit_temperature = False

data_folder = r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\\"
path_CO_temp = r"\\linelists\temp\\"
name_CO_temp = 'CO_temp'

plot_fits = False
save_fits = False

ig_start = 0 # start processing IG's at #ig_start
ig_stop = 69 # assume the process has completed itself by ig_stop 
ig_avg = 5 # how many to average together

ig_inc_shock = 19.5 # average location of the incident shock (this is what the data is clocked off of)
t_inc2ref_shock = 35 # time between incident and reflected shock in microseconds

fits_plot = ['temperature', 'pressure', 'molefraction', 'shift']


#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 

forderLP = 2
fcutoffLP = 0.15 # low pass filter for smoothing the vacuum scan

baseline_TD_start = 20
baseline_TD_stop = 0 # 0 is none, high numbers start removing high noise datapoints (doesn't seem to change much)

#%% -------------------------------------- load and smooth the vacuum scan -------------------------------------- 

files_vac = ['vacuum_background_21.npy', 'vacuum_background_23.npy']
trans_vacs_smooth = [None] * len(files_vac)

for i_file, file in enumerate(files_vac):

    IG_vac = np.load(data_folder + file)
    ppIG = len(IG_vac)
    
    i_center = int((ppIG-1)/2) + 1
    trans_vac = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_vac))).__abs__()[i_center:] # fft and remove reflected portion
    
    b, a = signal.butter(forderLP, fcutoffLP)
    trans_vacs_smooth[i_file] = signal.filtfilt(b, a, trans_vac)


#%% -------------------------------------- setup wavenumber axis based on lock conditions -------------------------------------- 

wvn_target = 2175 # a wavenumber we know is in our range
wvn2_fit = [2080, 2245]

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
wvn = np.arange(nyq_start, nyq_stop, favg)[1:] * hz2cm # convert from Hz to cm-1, remove last point so things match
wvl = 10000 / wvn

#%% -------------------------------------- generally applicable model conditions -------------------------------------- 

if co_argon_database: molecule_name = 'CO_Ar'
else: molecule_name = 'CO'

molecule_id = 5
PL = 1.27 # cm length of cell in furnace (double pass)
y_CO = 0.05

T_pre = 300 # temperature in K before the shock (for scaling trioxane measurement)
P_pre = 0.44 # pressure in atm before the shock (for scaling trioxane measurement)

# P_all =           [    3,      3,     5,     3,     3,      3,        5,       3,       3] # pressure in atm after shock (if assuming constant P)
T_all =           [ 1200,   1200,  1200,  1500,  1820,    1200,    1200,    1500,    1820]  # temperature in K
meas_file_names = ['1Ar', '1ArL', '2Ar', '3Ar', '4Ar', '1ArHe', '2ArHe', '3ArHe', '4ArHe']

#%% -------------------------------------- load HITRAN model -------------------------------------- 

df_CO = db.par_to_df(os.path.abspath('') + r'\linelists\\' + molecule_name + '.data')

df_CO = df_CO[(df_CO.nu > wvn2_fit[0]) & (df_CO.nu < wvn2_fit[1])] # wavenumber range
df_CO_fund = df_CO[df_CO.quanta.str.split(expand=True)[0] == '1'] # only looking at fundametal transitions for now

nu_delta = df_CO_fund.nu.to_list()[1] - df_CO_fund.nu.to_list()[0] # spacing between features

#%% -------------------------------------- setup for given file and load measurement data --------------------------------------   
    
fit_results_feature = {}
fit_results_global = {}

for i_file, meas_file in enumerate(meas_file_names): 

    if i_file in [1, 5,6,7,8]: i_vac = 0
    else: i_vac = 1
    
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
            
    fit_results_feature[meas_file] = np.zeros((len(ig_start_iters), len(df_CO_fund.nu), 2 + 2*len(fits_plot)))
    fit_results_global[meas_file] = np.zeros((len(ig_start_iters), 3 + 2*len(fits_plot)))
    
    for i_ig, ig_start_iter in enumerate(ig_start_iters): 
        
        ig_stop_iter = ig_start_iter + ig_avg
    
        print('*************************************************')
        print('****************** ' + meas_file + ' ******************')
        print('****** IG start:'+str(ig_start_iter)+ ' IG stop:' + str(ig_stop_iter-1)+' ******************')
        print('*************************************************')
    
        ig_avg_location = (ig_start_iter + ig_stop_iter - 1) / 2 - ig_inc_shock  # average full IG periodes post shock
        t_processing = ig_avg_location / dfrep * 1e6 - t_inc2ref_shock  # time referenced to the reflected Shock
        
        # temperature and pressure as f(time)
        P = pressure_data_P_smooth[np.argmin(abs(pressure_data_t-t_processing))] 
        
        if t_processing < 0: 
            T = T_pre # pre vs post shock temperature
        else: 
            T = T_all[i_file]
            
        # average IGs together
        IG_avg = np.mean(IG_all[ig_start_iter:ig_stop_iter,:],axis=0)
        
        meas_avg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_avg))).__abs__()[i_center:] # FFT and remove reflected portion
        
        # divide by vacuum to mostly normalize things
        trans_meas = meas_avg / trans_vacs_smooth[i_vac]
        
        # normalize max value to 1 (ish)
        i_target = np.argmin(abs(wvn-wvn_target))
        trans_meas = trans_meas / max(trans_meas[i_target-50:i_target+50])

        abs_meas = - np.log(trans_meas)
        
        #%% -------------------------------------- setup the model for fitting global temperature  -------------------------------------- 

        i_fits = td.bandwidth_select_td(wvn, wvn2_fit, max_prime_factor=50, print_value=False) # wavenumber indices of interest
        
        wvn_fit = wvn[i_fits[0]:i_fits[1]]
        abs_fit = abs_meas[i_fits[0]:i_fits[1]]
        
        TD_fit = np.fft.irfft(abs_fit) 
        
        pld.db_begin(r'linelists')  # load the linelists into Python        
    
        mod, pars = td.spectra_single_lmfit() 
        
        pars['mol_id'].value = molecule_id
        pars['shift'].vary = True
        pars['pathlength'].set(value = PL, vary = False)
        
        pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = fit_pressure)
        pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = True, min=200, max=3000)        
        pars['molefraction'].set(value = y_CO + y_CO*np.random.rand()/1000, vary = fit_concentration)
        
        weight = td.weight_func(len(abs_fit), baseline_TD_start, baseline_TD_stop)
        fit = mod.fit(TD_fit, xx = wvn_fit, params = pars, weights = weight, name=molecule_name)
        
        fit_results_global[meas_file][i_ig, 0] = t_processing
        
        for i_results, which_results in enumerate(fits_plot): 
                           
            # save some fit results and errors for plotting later
            fit_results_global[meas_file][i_ig, 2*i_results+1] = fit.params[which_results].value
            fit_results_global[meas_file][i_ig, 2*i_results+2] = fit.params[which_results].stderr
    
    #%% -------------------------------------- use the model to fit each feature one-by-one -------------------------------------- 
    
        for i_feature, nu_center in enumerate(df_CO_fund.nu):
    
            nu_left = nu_center-nu_delta/2
            nu_right = nu_center+nu_delta/2
            i_fits = td.bandwidth_select_td(wvn, [nu_left,nu_right], max_prime_factor=50, print_value=False) # wavenumber indices of interest

            wvn_fit = wvn[i_fits[0]:i_fits[1]]
            abs_fit = abs_meas[i_fits[0]:i_fits[1]]
            
            TD_fit = np.fft.irfft(abs_fit) 
            
            # shrink the model to only include region of interest
            df_CO_iter = df_CO[(df_CO.nu > nu_left - 0.5) & (df_CO.nu < nu_right + 0.5)]
            db.df_to_par(df_CO_iter.reset_index(), par_name=name_CO_temp, save_dir=os.path.abspath('')+path_CO_temp, print_name=False)
    
            pld.db_begin(r'linelists\temp')  # load the linelists into Python        
            
            T = fit_results_global[meas_file][i_ig, 2*fits_plot.index('temperature')+1] # fit temperature from whole spectra
            
            pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = fit_pressure)
            pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = fit_temperature, min=200, max=3000)        
            pars['molefraction'].set(value = y_CO + y_CO*np.random.rand()/1000, vary = fit_concentration)
    
            weight = td.weight_func(len(abs_fit), baseline_TD_start//10, baseline_TD_stop)

            fit = mod.fit(TD_fit, xx = wvn_fit, params = pars, weights = weight, name=name_CO_temp)
            
            fit_results_feature[meas_file][i_ig, i_feature, 0] = t_processing
            
            for i_results, which_results in enumerate(fits_plot): 
                               
                # save some fit results and errors for plotting later
                fit_results_feature[meas_file][i_ig, i_feature, 2*i_results+1] = fit.params[which_results].value
                fit_results_feature[meas_file][i_ig, i_feature, 2*i_results+2] = fit.params[which_results].stderr
                    
    #%% -------------------------------------- use model conditions to find integrated area for that feature -------------------------------------- 
            
            # shrink the model to only include single feature of interest
            db.df_to_par(df_CO_fund.iloc[[i_feature]].reset_index(), par_name=name_CO_temp, save_dir=os.path.abspath('')+path_CO_temp, print_name=False)

            pld.db_begin(r'linelists\temp')  # load the linelists into Python
            
            wvn_int = np.linspace(wvn_fit[0], wvn_fit[-1], 1000)
            TD_model_int = mod.eval(xx=wvn_int, params=fit.params, name=name_CO_temp)
            abs_model_int = np.real(np.fft.rfft(TD_model_int))
            
            fit_results_feature[meas_file][i_ig, i_feature, -1] = np.trapz(abs_model_int, wvn_int)
                
    #%% -------------------------------------- fit temperature using integrated area -------------------------------------- 
        
        def boltzman_strength(T, nu, sw, elower, c): 
            return lab.strength_T(T, elower, nu, molec_id=5) * sw * c
        
        mod_bolt = Model(boltzman_strength,independent_vars=['nu','sw','elower'])
        mod_bolt.set_param_hint('T',value=T, min=200, max=3000)
        mod_bolt.set_param_hint('c',value=1e20)

        
        result_bolt = mod_bolt.fit(fit_results_feature[meas_file][i_ig, :, -1], nu=df_CO_fund.nu, sw=df_CO_fund.sw, elower=df_CO_fund.elower)
        
        fit_results_global[meas_file][i_ig, -2] = result_bolt.params['T'].value
        fit_results_global[meas_file][i_ig, -1] = result_bolt.params['c'].value
        
        plt.figure()
        plt.plot(df_CO_fund.elower, fit_results_feature[meas_file][i_ig, :, -1])
        plt.plot(df_CO_fund.elower, result_bolt.best_fit)
        plt.title(i_ig)

# asdfsdfsd
        
        #%% -------------------------------------- plot stuff -------------------------------------- 


plt.figure()
meas_file = '1Ar'

for i_ig, ig_start_iter in enumerate(ig_start_iters):

    # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    # plt.title('{} for {} while averaging {} IGs.npy'.format(which_results, meas_file, ig_avg))                       
    
    gray = 0 # i_ig / len(ig_start_iters)
                
    plot_x = df_CO_fund.elower
    plot_y = fit_results_feature[meas_file][i_ig, :, -1]
            
    # plt.errorbar(plot_x, plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
    plt.plot(plot_x, plot_y, marker='x', label=meas_file , zorder=2, color=str(gray))
        
    plt.xlabel('Lower State Energy (E")')
    # plt.ylabel('{}'.format(which_results))
    
    # plt.legend(loc='lower right')


        #%% -------------------------------------- plot stuff -------------------------------------- 



plt.figure()
colors = ['tab:blue','tab:orange','tab:red']

for i_meas, meas_file in enumerate(['1Ar', '1ArL', '2Ar', '3Ar']):

    plt.plot(fit_results_global[meas_file][:,0], fit_results_global[meas_file][:,-2], linestyle='solid',
             label=meas_file+' boltzman', color=colors[i_meas])
    plt.plot(fit_results_global[meas_file][:,0], fit_results_global[meas_file][:, 2*fits_plot.index('temperature')+1], linestyle='dashed',
             label=meas_file+' global fit', color=colors[i_meas])
    plt.legend()








            


