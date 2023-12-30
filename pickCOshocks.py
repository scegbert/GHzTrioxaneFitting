#%% -------------------------------------- load some libraries -------------------------------------- 

# import time 
# time.sleep((60*60*2))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


#%% -------------------------------------- load the excel files from UIC -------------------------------------- 

folder_name = r"C:\Users\silmaril\Documents\from scott - making silmaril a water computer\GHzTrioxaneFitting\UIC data"
file_names = os.listdir(folder_name)

T5_limits = [[1206, 1247], 
             [1791, 1856],
             [1198, 1235],
             [1209, 1227],# [1190, 1227]
             [1476, 1537],
             [1202, 1231],
             [10, 1216],
             [1497, 1545],
             [1815, 1890]]

bin_spacing = [2, 25, 5]
test = False


for i, file in enumerate(file_names): 
    
    csv_name = os.path.join(folder_name, file, 'Surf_   ' + file.split()[-1][:-1].split('_')[-1] + '_WS.csv')
    df_full = pd.read_csv(csv_name) 
    
    columns = ['T5 / K', 'U5 / cm/s', 'P5 / Torr']
    
    df = df_full[columns]
    
    
    df_keep = df[(df['T5 / K'] > T5_limits[i][0])&(df['T5 / K'] < T5_limits[i][1])]
    T_avg = df_keep['T5 / K'].mean()
    T_std = df_keep['T5 / K'].std()
    T_perc = T_std / T_avg * 100
    
    P_avg = df_keep['P5 / Torr'].mean()
    P_std = df_keep['P5 / Torr'].std()
    P_perc = P_std / P_avg * 100
    
    n_keep = len(df_keep['P5 / Torr'])
    
    
    T_avg_t = df_full['T5 / K'].mean()
    T_std_t = df_full['T5 / K'].std()
    T_perc_t = T_std_t / T_avg_t * 100
    
    P_avg_t = df_full['P5 / Torr'].mean()
    P_std_t = df_full['P5 / Torr'].std()
    P_perc_t = P_std_t / P_avg_t * 100
    
    n_total = len(df_full['P5 / Torr'])
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
    fig.suptitle(u'Only {} shocks | T = {:.0f}±{:.0f} K ({:.2f}%) | P = {:.0f}±{:.0f} Torr ({:.2f}%)\n   All {} shocks | T = {:.0f}±{:.0f} K ({:.2f}%) | P = {:.0f}±{:.0f} Torr ({:.2f}%)'.format(
                        n_keep, T_avg, T_std, T_perc, P_avg, P_std, P_perc, 
                        n_total, T_avg_t, T_std_t, T_perc_t, P_avg_t, P_std_t, P_perc_t))
    
    # Iterate through each column and create a histogram in the corresponding subplot
    for j, column in enumerate(columns):
        
        min_val = df[column].min()
        max_val = df[column].max()
        bins = np.arange(min_val, max_val + bin_spacing[j], bin_spacing[j])
        
        if test: 
            df[column].plot(kind='hist', bins=1000, ax=axes[j], title=column)
            df_keep[column].plot(kind='hist', bins=1000, ax=axes[j], title=column)
        
        else:     
            df[column].plot(kind='hist', bins=bins, ax=axes[j], title=column)
            df_keep[column].plot(kind='hist', bins=bins, ax=axes[j], title=column, color='k')
        
        axes[j].set_xlabel(column)
        axes[j].set_ylabel('Count')
        axes[j].set_title('')
    
    plt.tight_layout()
    
    plt.show()
    











