#%% -------------------------------------- load some libraries -------------------------------------- 

# import time 
# time.sleep((60*60*2))

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

errors = 0

#%% -------------------------------------- inputs we change sometimes -------------------------------------- 

f_counter_n = 10007604.8 # nominal near 10 MHz reading of the counter

co_argon_database = True

data_folder = r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\\"
path_CO_temp = r"\\linelists\temp\\"

plot_fits = True
save_fits = True

ig_start = 30 # start processing IG's at #ig_start
ig_stop = 69 # assume the process has completed itself by ig_stop 
ig_avg = 5 # how many to average together
runnning_average = False 

ig_inc_shock = 19.5 # average location of the incident shock (this is what the data is clocked off of)
t_inc2ref_shock = 35 # time between incident and reflected shock in microseconds

fits_plot = ['temperature', 'pressure', 'molefraction', 'shift']

#%% -------------------------------------- inputs you probably don't want to change -------------------------------------- 

forderLP = 1
fcutoffLP = 0.05 # low pass filter for smoothing the vacuum scan

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

i_ceo_beat = 3991 # index where ceo frequency is in spectrum
wvn_target = 2175 # a wavenumber we know is in our range (ie Nyquist window)
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
PL = 1.27 # cm length of shok tube
y_CO = 0.05

T_pre = 300 # temperature in K before the shock (for scaling trioxane measurement)
P_pre = 0.5 # pressure in atm before the shock (for scaling trioxane measurement)

# P_all =           [    3,      3,     5,     3,     3,      3,        5,       3,       3] # pressure in atm after shock (if assuming constant P)
T_all =           [ 1200,   1200,  1200,  1500,  1820,    1200,    1200,    1500,    1820]  # temperature in K
meas_file_names = ['1Ar', '1ArL', '2Ar', '3Ar', '4Ar', '1ArHe', '2ArHe', '3ArHe', '4ArHe']
vac_shift =        [[0,0],  [4,2],[12,15],[7,10],[5,4],   [2,4],   [0,1],   [0,0],  [0,0]] # how many points to shift the vacuum scan to line up the etalons

P5_trans = [4.878003, 2.934214, 2.957536, 3.257460, 2.647630, 2.836331, 5.115660, 2.969282, 3.506466] # peak transducer pressure (t=0) in atm
P5_WS =    [4.924693, 3.053094, 2.912574, 3.134610, 2.815670, 2.898593, 4.947627, 2.804127, 2.948197] # NS relation pressure in atm
T5_WS =    [1215.853, 1224.546, 1507.679, 1817.843, 1214.615, 1215.704, 1203.173, 1519.719, 1844.844] # NS relation temperature in K

#%% -------------------------------------- load HITRAN model -------------------------------------- 

df_CO = db.par_to_df(os.path.abspath('') + r'\linelists\\' + molecule_name + '.data')

df_CO[['quanta_U', 'quanta_L', 'quanta_branch', 'quanta_J']] = df_CO['quanta'].str.split(expand=True) # separate out quantum assignments column
df_CO[['quanta_U', 'quanta_L', 'quanta_J']] = df_CO[['quanta_U', 'quanta_L', 'quanta_J']].astype(int)
df_CO['quanta_m'] = df_CO['quanta_J'].copy()
df_CO.loc[df_CO.quanta_branch == 'P', 'quanta_m'] = df_CO[df_CO.quanta_branch == 'P'].quanta_J * -1
df_CO.loc[df_CO.quanta_branch == 'R', 'quanta_m'] = df_CO[df_CO.quanta_branch == 'R'].quanta_J + 1

df_CO = df_CO[(df_CO.nu > wvn2_fit[0]) & (df_CO.nu < wvn2_fit[1])] # wavenumber range

nu_delta = np.mean(df_CO[df_CO.quanta_U == 1].nu.diff()) # spacing between features in fundamental

separation = nu_delta / 4
# which = (((df_CO.nu.diff() > separation*1.25)&(-df_CO.nu.diff(periods=-1) > separation*1.25))|(df_CO.quanta_U == 1))
# which = (df_CO.quanta_U == 1) # tried without and got noiser T than I had seen earlier. Trying again. 
# df_CO_fund = df_CO[which].sort_values(['quanta_U','quanta_m']) # only looking at fundametal transitions for now, sort for plotting
df_CO_fund = df_CO.copy()

#%% -------------------------------------- setup for given file and load measurement data --------------------------------------   
    
fit_results_feature = {}
fit_results_global = {}

trans_all = {}

for i_file, meas_file in enumerate(meas_file_names):     
        
    name_CO_temp = 'CO_{}_{}_{}'.format(meas_file, ig_avg, co_argon_database) 
    
    # i_file = 4
    # meas_file = meas_file_names[i_file]
    
    # print(meas_file)
    # time.sleep(5)
    
    if i_file in [1, 5,6,7,8]: i_vac = 0
    else: i_vac = 1
    
    # load time resolved pressure data
    pressure_data = np.loadtxt(r"E:\ShockTubeData\DATA_MATT_PATRICK_TRIP_2\CO\averaged CO shock tube data\Averaged Pressure Profile {} update.csv".format(meas_file), delimiter=',')
    pressure_data_P = pressure_data[:,1] / 1.013 # convert from bar to atm
    pressure_data_t = pressure_data[:,0] * 1e6 # convert to microseconds
    
    pressure_data_P_smooth = pressure_data_P.copy() # smooth out the ringing in the pressure sensor
    b, a = signal.butter(forderLP, fcutoffLP)
    pressure_data_P_smooth[np.argmin(abs(pressure_data_t))+1:] = signal.filtfilt(b, a, pressure_data_P[np.argmin(abs(pressure_data_t))+1:])
        
    IG_all = np.load(data_folder+meas_file+'.npy') 
    
#%% -------------------------------------- average IGs together as desired (loop it) -------------------------------------- 

    if runnning_average: 
        # program will loop through them like this (assuming bins_avg = 3, ig_start = 15): 15b1+15b2+15b3, 15b2+15b3+15b4, ..., 15b7+15b8+16b1, 15b8+16b1+16b3, ...
        ig_start_iters = np.arange(ig_start, ig_stop - ig_avg+2)

    else: 
        
        ig_start_iters = np.arange(ig_start, ig_stop - ig_avg+2, ig_avg)
                
    fit_results_feature[meas_file] = np.zeros((len(ig_start_iters), len(df_CO_fund.nu), 3 + 2*len(fits_plot)))
    fit_results_global[meas_file] = np.zeros((len(ig_start_iters), 6 + 2*len(fits_plot)))
    
    trans_all[meas_file] = [None] * (len(ig_start_iters)+1)
    
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
        
        # shift vacuum around to line up etalons in the data
        trans_vacs_rolled = np.concatenate((np.roll(trans_vacs_smooth[i_vac], vac_shift[i_file][0])[:i_ceo_beat], 
                                           np.roll(trans_vacs_smooth[i_vac], vac_shift[i_file][1])[i_ceo_beat:]))
        
        # divide by vacuum to mostly normalize things
        trans_meas = meas_avg / trans_vacs_rolled
        
        # normalize max value to 1 (ish)
        i_target = np.argmin(abs(wvn-wvn_target))
        trans_meas = trans_meas / max(trans_meas[i_target-50:i_target+50])
        
        trans_meas[3946] = 1 # remove the peak from the giant CEO beat spike
        
        trans_all[meas_file][0] = wvn
        trans_all[meas_file][i_ig+1] = trans_meas

        abs_meas = - np.log(trans_meas)
        
        # plt.figure(1)
        # plt.plot(meas_avg / meas_avg[i_target])
        # # plt.plot(trans_vacs_smooth[i_vac] / trans_vacs_smooth[i_vac][i_target])      
        # plt.plot(trans_vacs_rolled / trans_vacs_rolled[i_target])      
        # plt.plot(trans_meas - 0.2)
        
        #%% -------------------------------------- setup the model for fitting global temperature using HITRAN -------------------------------------- 

        i_fits = td.bandwidth_select_td(wvn, wvn2_fit, max_prime_factor=50, print_value=False) # wavenumber indices of interest
        
        wvn_fit_global = wvn[i_fits[0]:i_fits[1]]
        abs_fit = abs_meas[i_fits[0]:i_fits[1]]
        
        TD_fit_global = np.fft.irfft(abs_fit) 
        
        pld.db_begin(r'linelists')  # load the linelists into Python        
    
        mod, pars = td.spectra_single_lmfit() 
        
        pars['mol_id'].value = molecule_id
        pars['shift'].vary = True
        pars['pathlength'].set(value = PL, vary = False)
        
        pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = True, max=20)
        pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = True, min=200, max=3000)        
        pars['molefraction'].set(value = y_CO + y_CO*np.random.rand()/1000, vary = True, max=0.5)
        
        weight = td.weight_func(len(abs_fit), baseline_TD_start, baseline_TD_stop)
        fit_global = mod.fit(TD_fit_global, xx = wvn_fit_global, params = pars, weights = weight, name=molecule_name)
        
        fit_results_global[meas_file][i_ig, 0] = t_processing
        fit_results_global[meas_file][i_ig, 1] = P # pressure according to Matt
        
        for i_results, which_results in enumerate(fits_plot): 
                           
            # save some fit results and errors for plotting later
            fit_results_global[meas_file][i_ig, 2*i_results+2] = fit_global.params[which_results].value
            fit_results_global[meas_file][i_ig, 2*i_results+3] = fit_global.params[which_results].stderr
            
            
    #%% -------------------------------------- use the model to fit each feature one-by-one -------------------------------------- 
        
        P = fit_results_global[meas_file][i_ig, 2*fits_plot.index('pressure')+2] # fit temperature from whole spectra
        T = fit_results_global[meas_file][i_ig, 2*fits_plot.index('temperature')+2] # fit pressure from whole spectra
        y_CO = fit_results_global[meas_file][i_ig, 2*fits_plot.index('molefraction')+2] # fit y from whole spectra
        shift = fit_results_global[meas_file][i_ig, 2*fits_plot.index('shift')+2] # fit shift from whole spectra
        
        for i_feature, nu_center in enumerate(df_CO_fund.nu):
    
            nu_left = nu_center-separation
            nu_right = nu_center+separation
            i_fits = td.bandwidth_select_td(wvn, [nu_left,nu_right], max_prime_factor=50, print_value=False) # wavenumber indices of interest

            wvn_fit = wvn[i_fits[0]:i_fits[1]]
            abs_fit = abs_meas[i_fits[0]:i_fits[1]]
            
            TD_fit = np.fft.irfft(abs_fit) 
            
            # shrink the model to only include region of interest
            df_CO_iter = df_CO[(df_CO.nu > nu_left - 0.5) & (df_CO.nu < nu_right + 0.5)]
            db.df_to_par(df_CO_iter.reset_index(), par_name=name_CO_temp, save_dir=os.path.abspath('')+path_CO_temp, print_name=False)
            
            error = True
            while error == True:
                try: # was getting an intermittent error here when running the full code on multiple kernals. trying to circumvent
                    pld.db_begin(r'linelists\temp')  # load the linelists into Python      
                    error = False
                except: 
                    errors+=1000
            
            pars['pressure'].set(value = P, vary = True, min=P/2, max=P*2)
            pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = True, min=200, max=3000)        
            pars['molefraction'].set(value = y_CO, vary = True, min=y_CO/2, max=y_CO*2)  
            
            pars['shift'].set(value = shift, vary = True, min=shift-0.01, max=shift+0.01)  
    
            weight = td.weight_func(len(abs_fit), baseline_TD_start//10, baseline_TD_stop)

            error = True
            while error == True:
                try: # was getting an intermittent error here when running the full code on multiple kernals. trying to circumvent
                    fit = mod.fit(TD_fit, xx = wvn_fit, params = pars, weights = weight, name=name_CO_temp)     
                    error = False
                except: 
                    errors+=1          
            
            fit_results_feature[meas_file][i_ig, i_feature, 0] = t_processing
            
            for i_results, which_results in enumerate(fits_plot): 
                
                # save some fit results and errors for plotting later (for troubleshooting, ie pressure jumps to 500 would probably be wrong)
                fit_results_feature[meas_file][i_ig, i_feature, 2*i_results+1] = fit.params[which_results].value
                fit_results_feature[meas_file][i_ig, i_feature, 2*i_results+2] = fit.params[which_results].stderr
            
    #%% -------------------------------------- use model conditions to find integrated area for that feature -------------------------------------- 
            
            # shrink the model to only include single feature of interest
            db.df_to_par(df_CO_fund.iloc[[i_feature]].reset_index(), par_name=name_CO_temp, save_dir=os.path.abspath('')+path_CO_temp, print_name=False)

            error = True
            while error == True:
                try: # was getting an intermittent error here when running the full code on multiple kernals. trying to circumvent
                    pld.db_begin(r'linelists\temp')  # load the linelists into Python      
                    error = False
                except: 
                    errors+=1000
            
            wvn_int = np.linspace(wvn_fit[0], wvn_fit[-1], 1000)
            TD_model_int = mod.eval(xx=wvn_int, params=fit.params, name=name_CO_temp)
            abs_model_int = np.real(np.fft.rfft(TD_model_int))
            
            # plt.figure(3*i_ig)
            # plt.plot(wvn_int, abs_model_int)
            # plt.plot(wvn_fit, abs_fit)
            
            fit_results_feature[meas_file][i_ig, i_feature, -2] = np.trapz(abs_model_int, wvn_int)
            
            # we're going to noise-weight the fit so it doesn't get dominated by the noisy features at the edges
            fit_results_feature[meas_file][i_ig, i_feature, -1] = np.std(abs_fit - np.real(np.fft.rfft(fit.best_fit)))
            
            
    #%% -------------------------------------- fit temperature using integrated area -------------------------------------- 
        
        def boltzman_strength(T, nu, sw, elower, c): 
            return lab.strength_T(T, elower, nu, molec_id=5) * sw * c # unitless scaling factor c
        
        mod_bolt = Model(boltzman_strength,independent_vars=['nu','sw','elower'])
        mod_bolt.set_param_hint('T',value=T, min=200, max=4000)
        
        # use ideal gas approximation for c - proportional to P*y/T, 1e22 is empirical from this data
        c_IG = 1e22 * P*y_CO / T
        mod_bolt.set_param_hint('c',value=c_IG)
                
        strength_estimate = boltzman_strength(T, df_CO_fund.nu, df_CO_fund.sw, df_CO_fund.elower, c_IG)
        weight_noise = (max(fit_results_feature[meas_file][i_ig, :, -1]) - fit_results_feature[meas_file][i_ig, :, -1]) # <--- unfortunately, exponent is arbitrary, max sets range from 0-1
        weight_noise = weight_noise / max(weight_noise)
        weight_strength = strength_estimate.to_numpy()
        weight_strength = weight_strength / max(weight_strength)
        
        weight = (weight_noise + weight_strength) 
        weight = weight / max(weight)
        
        result_bolt = mod_bolt.fit(fit_results_feature[meas_file][i_ig, :, -2], 
                                   nu=df_CO_fund.nu, sw=df_CO_fund.sw, elower=df_CO_fund.elower, 
                                   weights=weight) # noise weighting (focus on the good ones)
        
        fit_results_global[meas_file][i_ig, -4] = result_bolt.params['T'].value
        fit_results_global[meas_file][i_ig, -3] = result_bolt.params['T'].stderr
        
        fit_results_global[meas_file][i_ig, -2] = result_bolt.params['c'].value
        fit_results_global[meas_file][i_ig, -1] = result_bolt.params['c'].stderr
        
        # plt.figure()
               
        # for q in [1,2]: 
            
        #     if q==1: mark='v'
        #     elif q==2: mark='^'
            
        #     which = (df_CO_fund.quanta_U==q)
            
        #     plt.plot(df_CO_fund[which].quanta_m, fit_results_feature[meas_file][i_ig, :, -2][which], 
        #               color='tab:green', marker=mark, linewidth=3, label='measurement')
        
        #     plt.plot(df_CO_fund[which].quanta_m, result_bolt.best_fit[which], 
        #               color='tab:blue', marker='x', label='T = {} K (boltzmann)'.format(int(result_bolt.params['T'].value)), linewidth=3)
            
        #     plt.plot(df_CO_fund[which].quanta_m, strength_estimate[which], 
        #               color='tab:orange', marker='x', label='T = {} K (HITRAN)'.format(int(T)), linewidth=3)

        #     plt.plot(df_CO_fund[which].quanta_m,  weight[which]/10,
        #               '.', color='black', label='weight/10', linewidth=1)
        
        #     if q == 1: plt.legend()

        # plt.title(ig_start_iter)
        # plt.xlabel('quantum number (m)')
        # plt.ylabel('integrated area of feature')
        
        
        plt.ylim((0,0.3))
        
        # plt.savefig(r'C:\\Users\\scott\\Downloads\\plots\\{} {} IG {}.jpg'.format(meas_file, name_CO_temp, ig_start_iter), 
        #             bbox_inches='tight')
        # plt.close()
                
           
#%%    Plot HITRAN spectra

        # TD_model_fit = fit_global.best_fit
        # weight = fit_global.weights
        
        # # plot frequency-domain fit
        # abs_meas_noBL = np.real(np.fft.rfft(TD_fit_global - (1-weight) * (TD_fit_global - TD_model_fit)))
        # abs_model = np.real(np.fft.rfft(TD_model_fit))
        # wvl_plot = 10000 / wvn_fit_global
        
        # # plot with residual
        # fig, axs = plt.subplots(2,4, sharex = 'col', sharey = 'row', figsize=(12, 4),
        #                         gridspec_kw={'height_ratios': [3,1], 'width_ratios': [5,1,1,1], 'hspace':0.015, 'wspace':0.005})
        
        # # title
        # t_plot = str(int(np.round(t_processing)))
        # T_plot = str(int(np.round(fit.params['temperature'].value,0)))
        # y_plot = str(np.round(fit.params['molefraction'].value*100,1)) 
        # P_plot = str(np.round(fit.params['pressure'].value,1))
        # shift_plot = str(np.round(fit.params['shift'].value,4))
        
        # plot_title = '{}, IG {} T_opt={}K P_opt={}atm'.format(meas_file, ig_start_iter, T_plot, P_plot)
        
        # plt.suptitle(plot_title)
        
        # # top first plot - absorbance over wavelength for both model and meas
        # axs[0,0].plot(wvl_plot, abs_meas_noBL, label='meas')
        # axs[0,0].plot(wvl_plot, abs_model, label='model')
        # axs[0,0].legend(loc='upper right')
        # axs[0,0].set_ylabel('Absorbance')
        
        # axs[0,0].set_ylim(-0.25, 1.5)

        # # bottom first plot - absorbance over wavelength for both model and meas        
        # axs[1,0].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
        # axs[1,0].legend(loc='upper right')
        # axs[1,0].set_ylabel('Residual')
        # axs[1,0].set_xlabel('Wavelength (um)')
        
        # axs[1,0].set_ylim(-0.4, 0.4)
                
        
        
        # # top second plot - absorbance over some wavelength for both model and meas
        # axs[0,1].plot(wvl_plot, abs_meas_noBL, label='meas')
        # axs[0,1].plot(wvl_plot, abs_model, label='model')
        # axs[0,1].set_xlim(4.482, 4.4862)
           
        # # bottom second plot - absorbance over some wavelength for both model and meas        
        # axs[1,1].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
        

        # # top third plot - absorbance over some wavelength for both model and meas
        # axs[0,2].plot(wvl_plot, abs_meas_noBL, label='meas')
        # axs[0,2].plot(wvl_plot, abs_model, label='model')
        # axs[0,2].set_xlim(4.5177, 4.5288)
           
        # # bottom third plot - absorbance over some wavelength for both model and meas        
        # axs[1,2].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
        # axs[1,2].set_xlabel('Wavelength (um)')
        
        # # top fourth plot - absorbance over some wavelength for both model and meas
        # axs[0,3].plot(wvl_plot, abs_meas_noBL, label='meas')
        # axs[0,3].plot(wvl_plot, abs_model, label='model')
        # axs[0,3].set_xlim(4.5485, 4.5560)
           
        # # bottom fourth plot - absorbance over some wavelength for both model and meas        
        # axs[1,3].plot(wvl_plot, abs_meas_noBL - abs_model, label='meas-model')
        
        # if save_fits: 
            
        #     plt.savefig(os.path.abspath('') + r'\plots\\{} {} IG {}.jpg'.format(meas_file, name_CO_temp, ig_start_iter), 
        #             bbox_inches='tight')
        
        # plt.close()

np.save('fit results for labfit - narrow', [fit_results_global, trans_all])



sdfsdfsdfsdfsdf

        #%% -------------------------------------- plot the boltzmann curve you just fit -------------------------------------- 


plt.figure()
meas_file = '1Ar'

for i_ig, ig_start_iter in enumerate(ig_start_iters):

    # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    # plt.title('{} for {} while averaging {} IGs.npy'.format(which_results, meas_file, ig_avg))                       
    
    gray = i_ig / len(ig_start_iters)
                
    plot_x = df_CO_fund.nu
    plot_y = fit_results_feature[meas_file][i_ig, :, -2]
            
    # plt.errorbar(plot_x, plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
    plt.plot(plot_x, plot_y, marker='x', label=meas_file , zorder=2, color=str(gray))
        
    plt.xlabel('Wavenumber (cm-1) as proxy for lower state energy')
    # plt.ylabel('{}'.format(which_results))
    
    # plt.legend(loc='lower right')


        #%% -------------------------------------- plot boltzmann and HITRAN temperatures -------------------------------------- 

plot_offset = 0

plt.figure(figsize=(15, 4))
colors = ['navy','darkslateblue','darkgreen','darkred','darkorange','blue','seagreen','lightcoral','goldenrod']

for i_file, meas_file in enumerate(meas_file_names):

    # if int(meas_file[0]) < 2.5: 

    plt.plot(fit_results_global[meas_file][:,0] + i_file*plot_offset, fit_results_global[meas_file][:,-4], linestyle='solid',
             label=meas_file+' boltzman', color=colors[i_file], linewidth=3)
    plt.errorbar(fit_results_global[meas_file][:,0] + i_file*plot_offset, fit_results_global[meas_file][:,-4], 
                  yerr=fit_results_global[meas_file][:,-3], color='k', ls='dotted', linewidth=0.3, zorder=1)
    
    plt.plot(fit_results_global[meas_file][:,0] + i_file*plot_offset*1.2, fit_results_global[meas_file][:, 2*fits_plot.index('temperature')+2], linestyle='dashed',
              color=colors[i_file], linewidth=3)

plt.legend(loc='upper left')
plt.xlabel('time post shock (us)')
plt.ylabel('Temperature (K)')
plt.title('Averaging {} IGs'.format(ig_avg))

plt.xlim((-70,730))

# plt.xlim((-340, 740))
# plt.ylim((240,2590))

        #%% -------------------------------------- plot concentration estimates -------------------------------------- 

plot_offset = 5

R = 1

plt.figure(figsize=(15, 4))

colors = ['navy','darkslateblue','darkgreen','darkred','darkorange','blue','seagreen','lightcoral','goldenrod']

for i_file, meas_file in enumerate(meas_file_names):
    
    x_plot = fit_results_global[meas_file][:,0]
    
    P_matt = fit_results_global[meas_file][:,1] # time resolved transducer pressure
    # P5_trans = peak transducer pressure
    # P5_WS = peak NS pressure
    
    P_opt = fit_results_global[meas_file][:, 2*fits_plot.index('pressure')+2]
    y_opt = fit_results_global[meas_file][:, 2*fits_plot.index('molefraction')+2]

    # y_plot1 = (y_opt * P_opt / P_matt) #/0.05    
    y_plot1 = (y_opt * P5_WS[i_file] / P_opt[np.argmin(abs(x_plot - 22))]) #/0.05
    y_plot2 = (y_opt) #/0.05
    
    y_plot3 = (y_plot2 - y_plot1) / y_plot1
    
    plt.axhline(0.05, color='k')
    
    plt.plot(x_plot, y_plot1, linestyle='solid', label=meas_file, color=colors[i_file], linewidth=3)
    # plt.plot(x_plot, y_plot2, linestyle='dashed', color=colors[i_file], linewidth=3)
    
    print('{}, {}, {}'.format(meas_file, np.mean(y_opt[np.argmin(abs(x_plot)):]), np.std(y_opt[np.argmin(abs(x_plot)):])))
        
plt.legend(loc='upper left')
plt.xlabel('time post reflected shock (us)')
plt.ylabel('Concentration')

plt.xlim((-70,730))
# plt.ylim((0.025, 0.15))
plt.ylim((0.031, 0.069))


        #%% -------------------------------------- plot pressure -------------------------------------- 

plot_offset = 5

R = 1

plt.figure(figsize=(15, 4))
colors = ['navy','darkslateblue','darkgreen','darkred','darkorange','blue','seagreen','lightcoral','goldenrod']

for i_file, meas_file in enumerate(meas_file_names):
    
    if meas_file[-1] == 'e': 
        
        x_plot = fit_results_global[meas_file][:,0]
        
        P_matt = fit_results_global[meas_file][:,1]
        # P_matt = P_opt * P_opt[np.argmin(abs(x_plot - 22))] / P5_trans[i_file]
        
        P_opt = fit_results_global[meas_file][:, 2*fits_plot.index('pressure')+2]
        
        y_plot = (P_opt - P_matt) / P_matt
               
        plt.plot(x_plot, P_opt, linestyle='solid', label=meas_file, color=colors[i_file], linewidth=3)
        plt.plot(x_plot, P_matt, linestyle='dashed', color=colors[i_file], linewidth=3)
        
    
plt.xlabel('time post reflected shock (us)')
# plt.ylabel('Pressure (opt-matt)/matt')
plt.ylabel('Pressure (atm)')
plt.legend(loc='upper left')

plt.xlim((-70,730))
plt.ylim((-0.1, 8.9))

                    #%% -------------------------------------- plot c value from boltzmann fits -------------------------------------- 

plot_offset = 5

R = 1

plt.figure(figsize=(15, 4))
colors = ['navy','darkslateblue','darkgreen','darkred','darkorange','blue','seagreen','lightcoral','goldenrod']

for i_file, meas_file in enumerate(meas_file_names):
    
    x_plot = fit_results_global[meas_file][:,0]
    
    y_plot = fit_results_global[meas_file][:,-2] * (fit_results_global[meas_file][:,-4] / fit_results_global[meas_file][:,1])
    
    y_plot = y_plot / 5.5e20 * 0.05
    
    # y_plot_err = fit_results_global[meas_file][:,-1]
    
    plt.plot(x_plot, y_plot, linestyle='solid', label=meas_file, color=colors[i_file], linewidth=3)

    # plt.errorbar(fit_results_global[meas_file][:,0] + i_file*plot_offset, fit_results_global[meas_file][:,-4], 
    #                   yerr=fit_results_global[meas_file][:,-3], color='k', ls='dotted', linewidth=0.3, zorder=1)

    
    
plt.xlabel('time post reflected shock (us)')
plt.ylabel('$C_{Boltzmann}$ (normalized)')
# plt.ylabel('Pressure (atm)')
plt.legend(loc='upper left')

plt.xlim((-70,730))
# plt.ylim((-0.1, 9.5))
plt.ylim((0.031, 0.069))

#%% -------------------------------------- average conditions over a range -------------------------------------- 


plt.figure()
colors = ['navy','darkslateblue','darkgreen','darkred','darkorange','blue','seagreen','lightcoral','goldenrod']

ig_start = 37

ig_step =15

ig_stop = ig_start + ig_step


for i_file, meas_file in enumerate(meas_file_names):
    
    if meas_file[-1] == 'e': 
        
        Pavg = np.mean(fit_results_global[meas_file][:,1][ig_start:ig_stop])
        Pstd = np.std(fit_results_global[meas_file][:,1][ig_start:ig_stop])
    
    
        Tavg = np.mean(fit_results_global[meas_file][:,-4][ig_start:ig_stop])
        Tstd = np.std(fit_results_global[meas_file][:,-4][ig_start:ig_stop])
        
        shift_avg = np.mean(fit_results_global[meas_file][:, 2*fits_plot.index('shift')+2][ig_start:ig_stop])
        shift_std = np.std(fit_results_global[meas_file][:, 2*fits_plot.index('shift')+2][ig_start:ig_stop])
    
        print('{}   P: {}   {}         T: {}   {}         shift: {}   {}'.format(meas_file, np.round(Pavg,4), np.round(Pstd,4), np.round(Tavg,4), np.round(Tstd,4)))
    
        plt.plot(ig_step, Pstd / Pavg, marker='x', label=meas_file, color=colors[i_file], linewidth=3)




