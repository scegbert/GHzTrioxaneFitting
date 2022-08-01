#%% -------------------------------------- load some libraries -------------------------------------- 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td # time doamain support
import linelist_conversions as db

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

from scipy.constants import speed_of_light
from scipy.optimize import curve_fit


#%% add quantum number stuff

def add_quantum_info(df): 

    if 'quanta_m' not in df.columns:
    
        df['quanta_m'] = df['quanta_J'].copy()
        df.loc[df.quanta_branch == 'P', 'quanta_m'] = df[df.quanta_branch == 'P'].quanta_J * -1
        df.loc[df.quanta_branch == 'R', 'quanta_m'] = df[df.quanta_branch == 'R'].quanta_J + 1
    
    df['quanta_UL'] = df['quanta_U']*10 + df['quanta_L']
    
    return df


#%% -------------------------------------- setup wavenumber axis that spans large range ---------------- 

d_database = r"C:\Users\scott\Documents\1-WorkStuff\code\GHzTrioxaneFitting\linelists\\"


wvn = np.arange(2150,2280,1e9/speed_of_light/100)
wvl = 10000 / wvn

PL = 91.4 # pathlength in cm
P = 16 / 760 # pressure in atm
T = 1100 # T in K
y = 1

molecule = 'CO'
molecule_id = 5


#%% -------------------------------------- separate the database by quantum vibrational assignments -------------------------------------- 


df_CO = db.par_to_df(d_database + molecule + '.data')
df_CO = df_CO[df_CO.local_iso_id == 1]

df_CO[['quanta_U', 'quanta_L', 'quanta_branch', 'quanta_J']] = df_CO['quanta'].str.split(expand=True) # separate out quantum assignments column

df_CO[['quanta_U', 'quanta_L', 'quanta_J']] = df_CO[['quanta_U', 'quanta_L', 'quanta_J']].astype(int)

df_CO = df_CO[((df_CO.quanta_L == 0) & (df_CO.quanta_U == 1)) | 
                         ((df_CO.quanta_L == 1) & (df_CO.quanta_U == 2))]

df_CO = add_quantum_info(df_CO)


#%% -------------------------------------- fit Keeyoon's equation to the HITRAN data -------------------------------------- 

# Sung et al, Intensities, collision-broadened half-widths, and collision-induced line shifts in the second overtone band of

def gamma_eq_air(m, c0, c1): 
    # gamma = (c0 + c1*|m|) / (|m| + N)
    # N = 12 for self (Sung), 8 for air (Sung), M = 2 for Ar (Kowzan)    
    return (c0 + c1*abs(m)) / (abs(m) + 8)

parameters, covariance = curve_fit(gamma_eq_air, df_CO.quanta_m,  df_CO.gamma_air)


#%% -------------------------------------- air plots -------------------------------------- 

properties = ['n_air', 'gamma_air']

branches = ['P','R', 'B'] # b for both
color = ['tab:blue', 'tab:orange', 'tab:green']

transitions_UL = [10, 21, 31]
size = [10, 7, 5]

df_all = [df_CO]
df_note = ['HITRAN']
marker = ['x', '+', '1', '3', 'o', 's']


for i_p, prop in enumerate(properties): 
        
    for i_d, df_iter in enumerate(df_all): 
               
        plt.figure(i_p)
        
        for i_t, transition_UL in enumerate(transitions_UL): 
            
            for i_b, branch in enumerate(branches): 
                            
                which = (df_iter.quanta_UL == transition_UL) & (df_iter.quanta_branch == branch)
                 
                if np.any(which) and prop in df_iter.columns: 
                    
                    plt.plot(abs(df_iter[which].quanta_m), df_iter[which][prop], linestyle='None', 
                             marker=marker[i_d], color=color[i_b], markersize=size[i_t], # fillstyle=fill[i_b],
                             label = '{} branch {}<-{} from {}'.format(branch, str(transition_UL)[0], str(transition_UL)[1], df_note[i_d]))
            
            plt.legend()
            plt.ylabel(prop)
            plt.xlabel('|m|')
    
    


#%% -------------------------------------- literature blast -------------------------------------- 

# data from Ren (Hanson) et al CO concentration and temperature sensor for...
# caveat: 
dict_ren = {'quanta_U':                 [    1,     2,     1,     1,     2], 
            'quanta_L':                 [    0,     1,     0,     0,     1],
            'quanta_branch':            [  'R',   'R',   'R',   'P',   'P'], 
            'quanta_J':                 [   12,    21,    13,    20,    14], 
            'gamma_Ar': [x / 2 for x in [0.079, 0.072, 0.079, 0.083, 0.074]], 
            'n_Ar':                     [0.581, 0.571, 0.600, 0.639, 0.560]}
df_ren = pd.DataFrame.from_dict(dict_ren, orient='columns')
df_ren = add_quantum_info(df_ren)


# data from Bendana (spearrin) et al Line mixing and broadening in the v(1→3) first overtone bandhead of carbon monoxide at high temperatures and high pressures
# caveat: wrong transition family, focused at higher temperatures
dict_bendana = {'quanta_U':      [    3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,    3,     3], 
                'quanta_L':      [    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1],
                'quanta_branch': [  'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',  'R',   'R'], 
                'quanta_J':      [   42,    43,    44,    45,    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,   57,    58], 
                'n_Ar':          [0.485, 0.455, 0.478, 0.425, 0.455, 0.429, 0.429, 0.422, 0.414, 0.406, 0.398, 0.390, 0.401, 0.393, 0.379, 0.359, 0.355]}
df_bendana = pd.DataFrame.from_dict(dict_bendana, orient='columns')
df_bendana = add_quantum_info(df_bendana)



# data from vanderover et al A mid-infrared scanned-wavelength laser absorption sensor for carbon monoxide and temperature measurements from 900 to 4000 K
# caveat: took data from Thibault, but they're not exactly the same...
dict_vanderover = {'quanta_U':      [     1,      1,      1,      1], 
                   'quanta_L':      [     0,      0,      0,      0],
                   'quanta_branch': [   'R',    'R',    'R',    'R'], 
                   'quanta_J':      [     9,     10,     17,     18], 
                   'gamma_Ar':      [0.0465, 0.0459, 0.0423, 0.0417], 
                   'n_Ar':          [0.720,  0.710,  0.699,  0.699]}
df_vanderover = pd.DataFrame.from_dict(dict_vanderover, orient='columns')
df_vanderover = add_quantum_info(df_vanderover)


# data from Thibault et al Raman and infrared linewidths of CO in Ar
# caveat: focused on 300-800 K
dict_thibault = {'quanta_U':                      [    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,      1,    1], 
                   'quanta_L':                    [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,      0,    0],
                   'quanta_branch':               [  'B',   'B',   'B',   'B',   'B',   'B',   'B',   'B',   'B',   'B',   'B',    'B',   'B',  'B'], 
                   'quanta_m':                    [    1,     2,     3,     4,     5,     6,     7,     8,     9,    10,    11,     13,    15,   17], 
                   'gamma_Ar': [x / 1000 for x in [71.25, 65.35,  60.5,  56.0,  52.6, 50.15,  48.5,  47.3, 46.45,  45.9,  45.4,  44.3,  43.5,  42.3]], 
                   'n_Ar':                        [0.766, 0.747, 0.723, 0.703, 0.700, 0.706, 0.717, 0.720, 0.721, 0.712, 0.718, 0.720, 0.713, 0.699]}
df_thibault = pd.DataFrame.from_dict(dict_thibault, orient='columns')
df_thibault = add_quantum_info(df_thibault)


# data from Kowzan et al, Fully quantum calculations of the line-shape parameters for the Hartmann-Tran profile: A CO-Ar case study
# caveat: parameters from full HTP, only two features, temperature dependence is a little hand-wavey
dict_kowzan = {'quanta_U':      [      1,       1], 
            'quanta_L':      [      0,       0],
            'quanta_branch': [    'P',     'P'], 
            'quanta_J':      [      2,       8], 
            'gamma_Ar':      [0.06568, 0.04790], 
            'n_Ar':          [  0.759,   0.710]}
df_kowzan = pd.DataFrame.from_dict(dict_kowzan, orient='columns')
df_kowzan = add_quantum_info(df_kowzan)


# data from Sinclair et al, Line Broadening in the Fundamental Band of CO in CO–He and CO–Ar Mixtures
# caveat: from 1997, only 300 K measurements
dict_sinclair = {'quanta_U':                    [    1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,    1,     1,     1,     1,     1,     1,     1,    1,  1], 
                 'quanta_L':                    [    0,     0,     0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0,    0,     0,     0,     0,     0,     0,     0,    0,  0],
                 'quanta_branch':               [  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',  'R',   'R',  'R',   'R',   'R',   'R',   'R',   'R',   'R',  'R','R'], 
                 'quanta_J':                    [    1,     2,     3,     5,     6,     7,     8,     9,   10,    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,    23,    25,    27,    27,     1,     2,     3,     4,     5,     6,     7,     8,     9,    10,   11,    12,   13,    14,    15,    16,    17,    18,    19,   20, 21], 
                 'gamma_Ar': [x / 1000 for x in [69.45, 64.16, 59.53, 52.42, 49.91, 48.14, 46.85, 45.83, 45.2, 44.48, 43.95, 43.44, 43.06, 42.70, 42.28, 41.84, 41.02, 40.45, 39.78, 37.35, 35.66, 33.63, 33.79, 69.58, 64.09, 59.55, 55.46, 51.94, 49.63, 47.89, 46.66, 45.56, 45.05, 44.2, 43.85, 43.4, 42.91, 42.48, 41.94, 41.49, 40.87, 40.42, 39.7, 39]]}
df_sinclair = pd.DataFrame.from_dict(dict_sinclair, orient='columns')
df_sinclair = add_quantum_info(df_sinclair)


#%% -------------------------------------- fit Keeyoon's equation to the Argon data -------------------------------------- 


def gamma_eq_Ar(m, c0, c1): 
    return (c0 + c1*abs(m)) / (abs(m) + 2)

parameters, covariance = curve_fit(gamma_eq_air, df_CO.quanta_m,  df_CO.gamma_air)


#%% -------------------------------------- Argon plots -------------------------------------- 

properties = ['n_Ar', 'gamma_Ar']

branches = ['P','R', 'B'] # b for both
color = ['tab:blue', 'tab:orange', 'tab:green']

transitions_UL = [10, 21, 31]
size = [10, 7, 5]

df_all = [df_ren, df_bendana, df_vanderover, df_thibault, df_kowzan, df_sinclair]
df_note = ['Ren', 'Bendana', 'Vandover', 'Thibault', 'Kowzan', 'Sinclair']
marker = ['x', '+', '1', '3', 'o', 's']


for i_p, prop in enumerate(properties): 
        
    for i_d, df_iter in enumerate(df_all): 
               
        plt.figure(i_p)
        
        for i_t, transition_UL in enumerate(transitions_UL): 
            
            for i_b, branch in enumerate(branches): 
                            
                which = (df_iter.quanta_UL == transition_UL) & (df_iter.quanta_branch == branch)
                 
                if np.any(which) and prop in df_iter.columns: 
                    
                    plt.plot(abs(df_iter[which].quanta_m), df_iter[which][prop], linestyle='None', 
                             marker=marker[i_d], color=color[i_b], markersize=size[i_t], # fillstyle=fill[i_b],
                             label = '{} branch {}<-{} from {}'.format(branch, str(transition_UL)[0], str(transition_UL)[1], df_note[i_d]))
            
            plt.legend()
            plt.ylabel(prop)
            plt.xlabel('|m|')
    


