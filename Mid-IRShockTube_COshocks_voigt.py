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


from lmfit import Model

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()


#%% -------------------------------------- inputs we change sometimes -------------------------------------- 

f_counter_n = 10007604.8 # nominal near 10 MHz reading of the counter

fit_pressure = True # <--------------- use wisely, probably need to update CO for argon broadening first 
time_resolved_pressure = True
co_argon_database = True

fit_concentration = True

data_folder = r"H:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\\"

plot_fits = False
save_fits = False

ig_start = 20 # start processing IG's at #ig_start
ig_stop = 69 # assume the process has completed itself by ig_stop 
ig_avg = 10 # how many to average together

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
wvn2_fit = [2145, 2214] # [2080, 2245]

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
T_all =           [ 1200,   1200,  1200,  1500,  1820,  1200,      1200,    1500,    1820]  # temperature in K
meas_file_names = ['1Ar', '1ArL', '2Ar', '3Ar', '4Ar', '1ArHe', '2ArHe', '3ArHe', '4ArHe']

#%% -------------------------------------- load model -------------------------------------- 

df_CO = db.par_to_df(os.path.abspath('') + r'\linelists\\' + molecule_name + '.data')

df_CO = df_CO[(df_CO.nu > wvn2_fit[0]) & (df_CO.nu < wvn2_fit[1])]
df_CO = df_CO[df_CO.quanta.str.split(expand=True)[0] == '1'] # only looking at fundametal transitions for now

#%% -------------------------------------- setup for given file and load measurement data -------------------------------------- 

fit_results = {}

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
    
    fit_results[meas_file] = np.zeros((len(ig_start_iters),1+2*len(fits_plot))) 
    
    for i_ig, ig_start_iter in enumerate(ig_start_iters): 
        
        ig_stop_iter = ig_start_iter + ig_avg
    
        print('*************************************************')
        print('****************** ' + meas_file + ' ******************')
        print('****** IG start:'+str(ig_start_iter)+ ' IG stop:' + str(ig_stop_iter-1)+' ******************')
        print('*************************************************')
    
        ig_avg_location = (ig_start_iter + ig_stop_iter - 1) / 2 - ig_inc_shock  # average full IG periodes post shock
        t_processing = ig_avg_location / dfrep * 1e6 - t_inc2ref_shock  # time referenced to the reflected Shock
        fit_results[meas_file][i_ig, 0] = t_processing

        if t_processing < 0: 
            T = T_pre # pre vs post shock temperature
        else: 
            T = T_all[i_file]

        
        if time_resolved_pressure: # if we want time resolved pressure
            P = pressure_data_P_smooth[np.argmin(abs(pressure_data_t-t_processing))] 
            
        # average IGs together
        IG_avg = np.mean(IG_all[ig_start_iter:ig_stop_iter,:],axis=0)
        
        meas_avg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(IG_avg))).__abs__()[i_center:] # FFT and remove reflected portion
        
        # divide by vacuum to mostly normalize things
        trans_meas = meas_avg / trans_vacs_smooth[i_vac]
        
        # normalize max value to 1 (ish)
        i_target = np.argmin(abs(wvn-wvn_target))
        trans_meas = trans_meas / max(trans_meas[i_target-50:i_target+50])

        abs_meas = - np.log(trans_meas)
        
        
        
        
        #%% -------------------------------------- fit features in measurements -------------------------------------- 
        
        
        df_CO.nu
        
        nu_delta = 2150.856008 - 2147.081134
        
        
        feature_index = 152
        nu_center = 2212.625365
 
        nu_left = nu_center-nu_delta/2
        nu_right = nu_center+nu_delta/2
        i_fits = td.bandwidth_select_td(wvn, [nu_left,nu_right], max_prime_factor=50) # wavenumber indices of interest
        
        abs_fit = abs_meas[i_fits[0]:i_fits[1]]
        wvn_fit = wvn[i_fits[0]:i_fits[1]]

    
        def Gaussian(x, x0, A, wG, zerolevel):
            return A* np.sqrt(np.log(2) / np.pi / wG**2) * np.exp(-(x-x0)**2*np.log(2) / wG**2) + zerolevel
        
        def Lorentzian(x, x0, A, wL, zerolevel):            
            return A* 1/np.pi * wL / (wL**2 + (x-x0)**2) + zerolevel
        
        def VoigtConv(x, x0, A, wG, wL, zerolevel):
            
            G = np.fft.fft(Gaussian(x,x0,1,wG,0))
            L = np.fft.fft(Lorentzian(x,x0,1,wL,0))
            V = np.real(np.fft.fftshift(np.fft.ifft(G*L)))
            
            return A* V/np.trapz(V,x) + zerolevel
        
        vModel = Model(VoigtConv)
        
        # general model parameters
        A_guess = 0.001
        zero_level_guess = np.mean(abs_fit)
        
        vModel.set_param_hint('x0', value = nu_center, min=nu_left, max=nu_right)
        vModel.set_param_hint('A', value = A_guess, min=0)
        vModel.set_param_hint('zerolevel', value = zero_level_guess)
        
        # doppler (thermal) parameters        
        Na = 6.02214129e26 # kmol-1, avogadro
        k = 1.380649e-23 # m2 kg s-2 K-1, boltzman
        M = 28.01 # kg kmol-1, mass of CO
        wG_guess = (nu_center * 100) / speed_of_light * np.sqrt(2*Na*k*T*np.log(2) / M) /10
        vModel.set_param_hint('wG', value = wG_guess, min=0)
        
        # voigt (collisional) parameters
        wL_guess = df_CO.gamma_air[feature_index] * P
        vModel.set_param_hint('wL', value = wL_guess, min=0)
        
        abs_fit_input = VoigtConv(wvn_fit, nu_center, A_guess, wG_guess, wL_guess, zero_level_guess)

        plt.plot(wvn_fit, abs_fit, label='measurement')
        plt.plot(wvn_fit, abs_fit_input, label='guess')
        
        result = vModel.fit(abs_fit, x=wvn_fit) #, method='nelder')
        
        print(result.best_values)
        
        rl = list(result.best_values.values())
    
        wvn_fit_fine2 = np.linspace(wvn_fit[0], wvn_fit[-1], 80)
        wvn_fit_fine3 = np.linspace(wvn_fit[0], wvn_fit[-1], 5000)
    
        abs_fit_result = VoigtConv(wvn_fit, *rl)
        abs_fit_result2 = VoigtConv(wvn_fit_fine2, *rl)
        abs_fit_result3 = VoigtConv(wvn_fit_fine3, *rl)

                
        plt.plot(wvn_fit, abs_fit_result, label='fit results')
        plt.plot(wvn_fit_fine2, abs_fit_result2, label='fit results2')
        plt.plot(wvn_fit_fine3, abs_fit_result3, label='fit results3')

        plt.legend()
        


        
        
        
        
        
        asdasdasd


        
