r'''

labfit1 - main water file

main file for processing things in labfit (calls labfit, sets up the files to be processed by the labfit fortran engine)


r'''



import subprocess

import numpy as np
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import linelist_conversions as db
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import labfithelp as lab

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

from copy import deepcopy

import pickle


# %% define some parameters

d_labfit = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - CO'
d_saved = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - CO\CO - saved'

bin_name = 'CO_Ar'
bins = {bin_name:[0, 2075, 2250, 0]}

d_old = os.path.join(d_labfit, bin_name, bin_name + '-000-og') # for comparing to original input files

ratio_min_plot = -2 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
offset = 60 # for plotting

cutoff_s296 = 5e-20



props = {}
props['nu'] = ['nu', 'ν', 1, 23, 0.0015] 
props['sw'] = ['sw', '$S_{296}$', 2, 24, 0.09] # 9 % percent
props['gamma_air'] = ['gamma_air', 'γ air', 3, 25, 0.012] 
props['elower'] = ['elower', 'E\"', 4, 34, 200] # only floating this when things are weird
props['n_air'] = ['n_air', 'n air', 5, 26, 0.13]
props['delta_air'] = ['delta_air', 'δ air', 6, 27, 0.005]
props['n_delta_air'] = ['n_delta_air', 'n δ air', 7, 28, 0.13]
props['MW'] = ['MW', 'MW', 8, 29, 1e6]
props['gamma_self'] = ['gamma_self', 'γ self', 9, 30, 0.10]
props['n_self'] = ['n_self', 'n γ self', 10, 31, 0.13]
props['delta_self'] = ['delta_self', 'δ self', 11, 32, 0.005]
props['n_delta_self'] = ['n_delta_self', 'n δ self', 12, 33, 0.13]
props['beta_g_self'] = ['beta_g_self', 'βg self', 13, 35, 1e6] # dicke narrowing (don't worry about it for water, can't float with SD anyway)
props['y_self'] = ['y_self', 'y self', 14, 36, 1e6] # rosenkrantz line mixing (don't worry about this one either)
props['sd_self'] = ['sd_self', 'speed dependence', 15, 37, 0.10] # pure and air
props[False] = False # used with props_which2 option (when there isn't a second prop)



#%% add all ASC files to the INP file and save original file

r'''
# lab.bin_ASC_cutoff(d_labfit, base_name, d_labfit, bins, bin_name, d_cutoff_locations, d_conditions)

lab.run_labfit(d_labfit, bin_name) # <-------------------

lab.save_file(d_labfit, bin_name, d_og=True) # make a folder for saving and save the original file for later
r'''



#%% float parameters and run Labfit


# [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit, bins, bin_name) # <-------------------
# df_calcs = lab.information_df(d_labfit, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------

# features = [int(x) for x in df_calcs[df_calcs.vpp==0].index.tolist() if (x>=10)&(x<=94)&(x not in [32., 34.])]
# features = [58, 60, 62, 64, 66, 68, 73, 75, 77, 79]
features = [62, 64, 66]


print('updating delta')
lab.float_lines(d_labfit, bin_name, features, props['delta_air'], 'rei_saved', []) # float lines, most recent saved REI in -> INP out
lab.run_labfit(d_labfit, bin_name, time_limit=10)

print('updating SW')
lab.float_lines(d_labfit, bin_name, features, props['sw'], 'rei_new', []) # float lines, most recent saved REI in -> INP out
lab.run_labfit(d_labfit, bin_name, time_limit=10)

# print('updating n_gamma')
# lab.float_lines(d_labfit, bin_name, features, props['n_air'], 'rei_new', []) # float lines, most recent saved REI in -> INP out
# lab.run_labfit(d_labfit, bin_name, time_limit=10)



[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props['delta_air'], axis_labels=False, doublets=False) # <-------------------

sdfsdf
#%% update feature widths

prop_which = 'n_air'
prop_which2 = 'delta_air'

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False, doublets=False) # <-------------------











