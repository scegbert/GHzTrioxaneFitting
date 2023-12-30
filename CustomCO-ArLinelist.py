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
    df['quanta_m_abs'] = abs(df['quanta_m'])
    
    return df

def Pade_22(m, c0, c1, c2, c3, d1, d2, d3): 
    return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2)

def Pade_33(m, c0, c1, c2, c3, d1, d2, d3): 
    return (c0 + c1*abs(m) + c2*abs(m)**2  + c3*abs(m)**3) / (1 + d1*abs(m) + d2*abs(m)**2 + d3*abs(m)**3)

def Pade_23(m, c0, c1, c2, d1, d2, d3): 
    return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2 + d3*abs(m)**3)

def Pade_32(m, c0, c1, c2, c3, d1, d2): 
    return (c0 + c1*abs(m) + c2*abs(m)**2  + c3*abs(m)**3) / (1 + d1*abs(m) + d2*abs(m)**2)

def const_then_slope(x, x1, y1, m1): 
    
    x = np.array(x)
    
    y = np.ones_like(x) * y1 # constant value portion
        
    y[x>x1] = m1*(x[x>x1] - x1) + y1 # linear portion, cntinuous with previous portion
    
    return y


def plot_fit(df_all, df_name, props, props_name, fit_all, fit_names, fit_x, transitions_UL): 

    plt.figure()
    
    prop_x = props[0]
    prop_y = props[1]
    
    # designations for branches 
    branches = ['P','R', 'B'] # b for both
    marker = ['x', '+', '1']
    
    # size for different branches (generally assumed [10, 21, 31], but maybe not all 3)
    size = [10, 7, 5]
    
    # colors for different sources (df's)
    color = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:brown']
    
    for i_d, df_iter in enumerate(df_all): 
                
        for i_t, transition_UL in enumerate(transitions_UL): 
            
            for i_b, branch in enumerate(branches): 
                            
                which = (df_iter.quanta_UL == transition_UL) & (df_iter.quanta_branch == branch)
                 
                if np.any(which) and prop_y in df_iter.columns: 
                    
                    plt.plot(df_iter[which][prop_x], df_iter[which][prop_y], linestyle='None', 
                             marker=marker[i_b], color=color[i_d], markersize=size[i_t], markeredgewidth=2,
                             label = '{} branch {}<-{} from {} ({})'.format(
                                 branch, str(transition_UL)[0], str(transition_UL)[1], df_name[i_d], which.sum()))        

    for i_f, fit_y in enumerate(fit_all): 
            
        plt.plot(fit_x[i_f], fit_y, color[i_f], label=fit_names[i_f])


    plt.legend()
    plt.xlabel(props_name[0])
    plt.ylabel(props_name[1])



#%% -------------------------------------- setup wavenumber axis that spans large range ---------------- 

d_database = r"C:\Users\silmaril\Documents\from scott - making silmaril a water computer\GHzTrioxaneFitting\linelists\\"


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

def gamma_Sung_air(m, c0, c1, N): 
    # N = 12 for self (Sung), 8 for air (Sung), M = 2 for Ar (Kowzan)    
    return (c0 + c1*abs(m)) / (abs(m) + N)

param_bounds=([-np.inf,-np.inf, 0],[np.inf,np.inf,100])

parameters_air_Sung, _ = curve_fit(gamma_Sung_air, df_CO.quanta_m,  df_CO.gamma_air, bounds = param_bounds)

m_fit_air = np.linspace(0,max(abs(df_CO.quanta_m))*1.1,1000)
gamma_air_Sung_fit = gamma_Sung_air(m_fit_air, parameters_air_Sung[0], parameters_air_Sung[1], parameters_air_Sung[2])

gamma_air_Sung = gamma_Sung_air(m_fit_air, 0.6717, 0.0381, 8)

#%% -------------------------------------- fit pade to HITRAN shift for 2->1 transition (smoother) -------------------------------------- 


df_CO_delta = df_CO[(df_CO.quanta_branch=='R') & (df_CO.quanta_m>=19) & (df_CO.quanta_UL==21)]

parameters_delta_air_Pade_22, _ = curve_fit(Pade_22, df_CO_delta.quanta_m,  df_CO_delta.delta_air)
delta_air_Pade_22 = Pade_22(df_CO_delta.quanta_m, *parameters_delta_air_Pade_22)

plt.plot(df_CO_delta.quanta_m, delta_air_Pade_22)



#%% -------------------------------------- air plots -------------------------------------- 

# properties = ['gamma_air', 'n_air', 'delta_air']

# branches = ['P','R', 'B'] # b for both
# marker = ['x', '+', '1', '3', 'o', 's']


# transitions_UL = [10, 21, 31]
# size = [10, 7, 5]

# df_all = [df_CO]
# df_note = ['HITRAN']
# color = ['tab:blue', 'tab:orange', 'tab:green']

# for i_p, prop in enumerate(properties): 
        
#     for i_d, df_iter in enumerate(df_all): 
               
#         plt.figure()
        
#         for i_t, transition_UL in enumerate(transitions_UL): 
            
#             for i_b, branch in enumerate(branches): 
                            
#                 which = (df_iter.quanta_UL == transition_UL) & (df_iter.quanta_branch == branch)
                 
#                 if np.any(which) and prop in df_iter.columns: 
                    
#                     if prop == 'delta_air': 
#                         plotx = df_iter[which].quanta_m
#                         plt.xlabel('m')

#                     else: 
#                         plotx = abs(df_iter[which].quanta_m)
#                         plt.xlabel('|m|')

#                     plt.plot(plotx, df_iter[which][prop], linestyle='None', 
#                              marker=marker[i_b], color=color[i_b], markersize=size[i_t], # fillstyle=fill[i_b],
#                              label = '{} branch {}<-{} from {}'.format(branch, str(transition_UL)[0], str(transition_UL)[1], df_note[i_d]))
            
#     if prop == 'gamma_air_nothing': 
        
#         # plt.plot(m_fit_air, gamma_air_poly_fit, 'g', label='polynomial fit')
        
#         plt.plot(m_fit_air, gamma_air_Pade_fit, 'b', label='Pade fit ({} terms)'.format(len(parameters_air_Pade)))
        
#         plt.plot(m_fit_air, gamma_air_Pade_fit, 'r', label='Pade fit (all)'.format(len(parameters_air_Pade)))

        
#         # plt.plot(m_fit_air, gamma_air_Sung, 'r', label='Sung Parameters (N=8)')
#         plt.plot(m_fit_air, gamma_air_Sung_fit, 'k', label='Sung Eq. Fit (N={})'.format(np.round(parameters_air_Sung[2],1)))

        
#     plt.legend()
#     plt.ylabel(prop)


#%% -------------------------------------- literature blast -------------------------------------- 


# df's with delta: 


# data from Ren (Hanson) et al CO concentration and temperature sensor for...
# caveat: widths fit in 1100-2000 region
dict_ren = {'quanta_U':                 [    1,     2,     1,     1,     2], 
            'quanta_L':                 [    0,     1,     0,     0,     1],
            'quanta_branch':            [  'R',   'R',   'R',   'P',   'P'], 
            'quanta_J':                 [   12,    21,    13,    20,    14], 
            'gamma_Ar': [x / 2 for x in [0.079, 0.072, 0.079, 0.083, 0.074]], 
            'n_Ar':                     [0.581, 0.571, 0.600, 0.639, 0.560]}
df_ren = pd.DataFrame.from_dict(dict_ren, orient='columns')
df_ren = add_quantum_info(df_ren)

# listed as 296 K measurement
dict_ren296 = {'quanta_U':                 [    1,     1,     1], 
               'quanta_L':                 [    0,     0,     0],
               'quanta_branch':            [  'R',   'R',   'P'], 
               'quanta_J':                 [   12,    13,    20], 
               'gamma_Ar': [x / 2 for x in [0.088, 0.085, 0.079]]}
df_ren296 = pd.DataFrame.from_dict(dict_ren296, orient='columns')
df_ren296 = add_quantum_info(df_ren296)



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
dict_kowzan = {'quanta_U':                 [      1,       1], 
            'quanta_L':                    [      0,       0],
            'quanta_branch':               [    'P',     'P'], 
            'quanta_J':                    [      2,       8], 
            'gamma_Ar':                    [0.06568, 0.04790], 
            'n_Ar':                        [  0.759,   0.710], 
            'delta_Ar': [x / 1000 for x in [  -1.48,   -3.55]]}
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

# data from Bouanich et al Linewidths of carbon monoxide self-broadening and broadened by argon and nitrogen
# caveat: way old
dict_bouanich = {'quanta_U':                    [  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,  1,    1,    1,    1,  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1], 
                 'quanta_L':                    [  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  0,    0,    0,    0,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
                 'quanta_branch':               ['B',  'B',  'B',  'B',  'B',  'B',  'B',  'B',  'B',  'B','B',  'B',  'B',  'B','B',  'B',  'B',  'B',  'B',  'B',  'B',  'B',  'B',  'B',  'B'], 
                 'quanta_m':                    [  1,    2,    3,    4,    5,    6,    7,    8,    9,   10, 11,   12,   13,   14, 15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25], 
                 'gamma_Ar': [x / 1000 for x in [ 66, 60.8, 57.3, 54.1, 51.9, 49.7, 48.3, 47.2, 46.4, 45.6, 45, 44.6, 44.2, 43.6, 43, 42.4, 41.7, 40.9, 40.2, 39.5, 38.8, 38.2, 37.7, 37.2, 36.6]]} 
df_bouanich = pd.DataFrame.from_dict(dict_bouanich, orient='columns')
df_bouanich = add_quantum_info(df_bouanich)

dict_bouanich_shift = {'quanta_U':               [    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,    1,     1,    1,     1,     1,     1,     1,     1], 
                  'quanta_L':                    [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0,    0,     0,    0,     0,     0,     0,     0,     0],
                  'quanta_branch':               [  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',  'P',   'P',   'P',   'P',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',  'R',   'R',  'R',   'R',  'R',   'R',   'R',   'R',   'R',   'R'], 
                  'quanta_J':                    [   23,    22,    21,    20,    19,    18,    17,    16,    15,    14,    13,   12,    11,    10,     9,     8,     7,     6,     5,    4,     3,     2,     1,     0,     1,     2,     3,     4,     5,     6,     7,     8,    9,    10,   11,    12,   16,    17,    19,    21,    22,    23], 
                  'delta_Ar': [x / 1000 for x in [-3.17, -3.06, -2.77, -3.12, -2.76, -2.72, -3.06, -3.09, -2.99, -3.01, -3.09, -2.9, -2.91, -2.94, -2.64, -2.91, -3.04, -3.17, -2.45, -2.5, -2.01, -1.31, -1.23, -1.14, -1.55, -1.56, -1.29, -1.51, -1.57, -1.85, -1.46, -1.59, -1.6, -1.75, -1.6, -1.71, -2.3, -2.16, -2.18, -2.48, -2.64, -3.08]]}
df_bouanich_shift = pd.DataFrame.from_dict(dict_bouanich_shift, orient='columns')
df_bouanich_shift = add_quantum_info(df_bouanich_shift)


# data from Luo et al, Shifting and broadening in the fundamental band of CO highly diluted in He and Ar: A comparison with theory
# caveat: 
dict_luo = {'quanta_U':                    [    1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1], 
            'quanta_L':                    [    0,     0,     0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
            'quanta_branch':               [  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',   'R',  'R',    'R',   'R',   'R',   'R'], 
            'quanta_J':                    [   21,    19,    17,    16,    15,    14,    13,    11,     9,     8,     7,     6,     5,     4,     3,    2,     1,    21,    19,    17,    16,    15,    13,    11,     9,     8,     7,     5,     4,     3,     2,     1], 
            'gamma_Ar': [x / 1000 for x in [38.79, 40.37, 41.67, 42.28, 42.92, 43.27, 43.72, 44.64, 45.87, 46.96, 48.05, 49.65, 52.08, 55.41, 59.42, 64.1, 69.71, 38.52, 40.31, 41.67, 42.24, 42.71, 43.64, 44.61, 45.84, 46.65, 47.92, 51.94, 55.31, 59.47, 64.18, 69.86]]}
df_luo = pd.DataFrame.from_dict(dict_luo, orient='columns')
df_luo = add_quantum_info(df_luo)

dict_luo_shift = {'quanta_U':                    [    1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1], 
                  'quanta_L':                    [    0,     0,     0,     0,     0,     0,     0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    0],
                  'quanta_branch':               [  'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'P',   'R',   'R',  'R',   'R',   'R',   'R',   'R',  'R',   'R',   'R',  'R',    'R',   'R',   'R',   'R',   'R'], 
                  'quanta_J':                    [   21,    19,    17,    16,    15,    14,    13,    11,     9,     8,     7,     6,     5,     4,     3,     2,     1,    24,    21,   19,    17,    16,    15,    13,   11,     9,     8,     7,     5,     4,     3,     2,     1], 
                  'delta_Ar': [x / 1000 for x in [-3.33, -3.35, -3.41, -3.43, -3.33, -3.41, -3.38, -3.32, -3.26, -3.24, -3.37, -3.24, -3.02, -2.72, -2.21, -1.39, -1.32, -2.95, -2.69, -2.3, -2.04, -1.91, -1.85, -1.72, -1.7, -1.42, -1.37, -1.28, -1.28, -1.21, -1.26, -1.53, -1.28]]}
df_luo_shift = pd.DataFrame.from_dict(dict_luo_shift, orient='columns')
df_luo_shift = add_quantum_info(df_luo_shift)





#%% -------------------------------------- fit Keeyoon's equation to the Argon data -------------------------------------- 

def gamma_Sung_Ar(m, c0, c1, N): 
    # N = 12 for self (Sung), 8 for air (Sung), M = 2 for Ar (Kowzan)    
    return (c0 + c1*abs(m)) / (abs(m) + N)



df_Ar = pd.concat([df_ren296, df_bendana, df_vanderover, df_thibault, df_kowzan, df_sinclair, df_bouanich, df_luo])
df_Ar_gamma = df_Ar[df_Ar['gamma_Ar'].notna()]

param_bounds=([-np.inf,-np.inf, 0],[np.inf,np.inf,np.inf])

parameters_Ar_Sung, _ = curve_fit(gamma_Sung_Ar, df_Ar_gamma.quanta_m,  df_Ar_gamma.gamma_Ar, bounds=param_bounds)

m_fit_Ar = np.linspace(0,40,1000)

gamma_Ar_Sung_fit = gamma_Sung_Ar(m_fit_Ar, parameters_Ar_Sung[0], parameters_Ar_Sung[1], parameters_Ar_Sung[2])


#%% -------------------------------------- Pade approximation for width -------------------------------------- 

parameters_Ar_Pade_22, _ = curve_fit(Pade_22, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.gamma_Ar)
gamma_Ar_Pade_fit_22 = Pade_22(m_fit_Ar, *parameters_Ar_Pade_22)

parameters_Ar_Pade_33, _ = curve_fit(Pade_33, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.gamma_Ar)
gamma_Ar_Pade_fit_33 = Pade_33(m_fit_Ar, *parameters_Ar_Pade_33)

parameters_Ar_Pade_23, _ = curve_fit(Pade_23, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.gamma_Ar)
gamma_Ar_Pade_fit_23 = Pade_23(m_fit_Ar, *parameters_Ar_Pade_23)

parameters_Ar_Pade_32, _ = curve_fit(Pade_32, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.gamma_Ar)
gamma_Ar_Pade_fit_32 = Pade_32(m_fit_Ar, *parameters_Ar_Pade_32)



#%% -------------------------------------- Argon width plots -------------------------------------- 


# df's with gamma: ren, ren296, vanderover, thibault, kowzan, sinclair, bouanich, luo


df_all = [df_sinclair, df_thibault,  df_bouanich, df_luo, df_bendana, df_vanderover, df_kowzan, df_ren296]
df_name = [ 'Sinclair', 'Thibault', 'Bouanich', 'Luo', 'Bendana', 'Vandover', 'Kowzan', 'Ren @ 296']

props = ['quanta_m_abs', 'gamma_Ar'] # x, y
props_name = ['|m|', 'gamma Argon']

fit_all = [gamma_Ar_Pade_fit_32, gamma_Ar_Pade_fit_33, gamma_Ar_Pade_fit_23]
fit_names = ['Pade 3/2', 'Pade 3/3', 'Pade 2/3']

fit_x = m_fit_Ar

transitions_UL = [10, 21, 31]

# plot_fit(df_all, df_name, props, props_name, fit_all, fit_names, fit_x, transitions_UL)
    

#%% -------------------------------------- Pade approximation for temperature dependence -------------------------------------- 


df_Ar = pd.concat([df_bendana, df_vanderover, df_thibault, df_kowzan])
df_Ar_gamma = df_Ar[df_Ar['n_Ar'].notna()]


n_parameters_Ar_Pade_33, _ = curve_fit(Pade_33, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.n_Ar)
n_Ar_Pade_fit_33 = Pade_33(m_fit_Ar, *n_parameters_Ar_Pade_33)

n_parameters_Ar_Pade_22, _ = curve_fit(Pade_22, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.n_Ar)
n_Ar_Pade_fit_22 = Pade_22(m_fit_Ar, *n_parameters_Ar_Pade_22)

n_parameters_Ar_Pade_32, _ = curve_fit(Pade_32, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.n_Ar)
n_Ar_Pade_fit_32 = Pade_32(m_fit_Ar, *n_parameters_Ar_Pade_32)

n_parameters_Ar_Pade_23, _ = curve_fit(Pade_23, df_Ar_gamma.quanta_m_abs,  df_Ar_gamma.n_Ar)
n_Ar_Pade_fit_23 = Pade_23(m_fit_Ar, *n_parameters_Ar_Pade_23)


    

#%% -------------------------------------- Argon temperature dependence of the width plots -------------------------------------- 

# df's with n_gamma: ren, bendana*, vanderover, thibault, kowzan, 

df_all = [df_sinclair, df_thibault,  df_bouanich, df_luo, df_bendana, df_vanderover, df_kowzan, df_ren296]
df_name = [ 'Sinclair', 'Thibault', 'Bouanich', 'Luo', 'Bendana', 'Vandover', 'Kowzan', 'Ren @ 296']

props = ['quanta_m_abs', 'gamma_Ar'] # x, y
props_name = ['|m|', 'gamma Argon']

fit_all = [gamma_Ar_Pade_fit_32, gamma_Ar_Pade_fit_33, gamma_Ar_Pade_fit_23]
fit_names = ['Pade 3/2', 'Pade 3/3', 'Pade 2/3']

fit_x = m_fit_Ar

transitions_UL = [10, 21, 31]

# plot_fit(df_all, df_name, props, props_name, fit_all, fit_names, fit_x, transitions_UL)




#%% -------------------------------------- Pade approximation for shift -------------------------------------- 


df_Ar = pd.concat([df_luo_shift, df_kowzan,  df_bouanich_shift])
df_Ar_delta = df_Ar[df_Ar['delta_Ar'].notna()]
df_Ar_deltaR = df_Ar_delta[df_Ar_delta['quanta_branch'] == 'R']
df_Ar_deltaP = df_Ar_delta[df_Ar_delta['quanta_branch'] == 'P']

m_fit_Ar_deltaR = np.linspace(0,40,1000)
m_fit_Ar_deltaP = np.linspace(-40,0,1000)

delta_parameters_Ar_Pade_33, _ = curve_fit(Pade_33, df_Ar_deltaR.quanta_m,  df_Ar_deltaR.delta_Ar)
delta_Ar_Pade_fit_33 = Pade_33(m_fit_Ar_deltaR, *delta_parameters_Ar_Pade_33)

delta_parameters_Ar_Pade_22, _ = curve_fit(Pade_22, df_Ar_deltaR.quanta_m,  df_Ar_deltaR.delta_Ar)
delta_Ar_Pade_fit_22 = Pade_22(m_fit_Ar_deltaR, *delta_parameters_Ar_Pade_22)

delta_parameters_Ar_Pade_32, _ = curve_fit(Pade_32, df_Ar_deltaR.quanta_m,  df_Ar_deltaR.delta_Ar)
delta_Ar_Pade_fit_32 = Pade_32(m_fit_Ar_deltaR, *delta_parameters_Ar_Pade_32)

delta_parameters_Ar_Pade_23, _ = curve_fit(Pade_23, df_Ar_deltaR.quanta_m,  df_Ar_deltaR.delta_Ar)
delta_Ar_Pade_fit_23 = Pade_23(m_fit_Ar_deltaR, *delta_parameters_Ar_Pade_23)

delta_parameters_Ar_poly2R = np.polyfit(df_Ar_deltaR.quanta_m,  df_Ar_deltaR.delta_Ar, 2)
delta_Ar_poly2R = np.poly1d(delta_parameters_Ar_poly2R)(m_fit_Ar_deltaR)

delta_parameters_Ar_poly2P = np.polyfit(df_Ar_deltaP.quanta_m,  df_Ar_deltaP.delta_Ar, 3)
delta_Ar_poly2P = np.poly1d(delta_parameters_Ar_poly2P)(m_fit_Ar_deltaP)

guess = [-7, -0.003, 0.0002]
delta_parameters_Ar_CtSP, _ = curve_fit(const_then_slope, df_Ar_deltaP.quanta_m,  df_Ar_deltaP.delta_Ar, guess) 
delta_Ar_CtSP = const_then_slope(m_fit_Ar_deltaP, *delta_parameters_Ar_CtSP)


#%% -------------------------------------- Argon shift plots -------------------------------------- 

# df's with shift_Ar: luo_shift, kowzan, bouanich_shift

df_all = [df_luo_shift, df_kowzan,  df_bouanich_shift]
df_name = [ 'Luo', 'Kowzan', 'Bouanich']

props = ['quanta_m', 'delta_Ar'] # x, y
props_name = ['m', 'delta Argon']

fit_all = [delta_Ar_poly2R, delta_Ar_CtSP]
fit_names = ['2nd order poly R branch', 'constant then linear']

fit_x = [m_fit_Ar_deltaR, m_fit_Ar_deltaP] 

transitions_UL = [10, 21, 31]

# plot_fit(df_all, df_name, props, props_name, fit_all, fit_names, fit_x, transitions_UL)



#%% -------------------------------------- change HITRAN values to updated Argon values and save as data file -------------------------------------- 


df_Ar_HITRAN = df_CO.copy()

df_Ar_HITRAN.gamma_air = Pade_23(df_Ar_HITRAN.quanta_m_abs, *parameters_Ar_Pade_23)
df_Ar_HITRAN.n_air = Pade_23(df_Ar_HITRAN.quanta_m_abs, *n_parameters_Ar_Pade_23)

# df_Ar_HITRAN.delta_air = 0

# only changing things tp fit equation for R branch m<20
df_Ar_HITRAN.loc[(df_Ar_HITRAN.quanta_branch=='R') & (df_Ar_HITRAN.quanta_m<20), 'delta_air'] = np.poly1d(
    delta_parameters_Ar_poly2R)(df_Ar_HITRAN[(df_Ar_HITRAN.quanta_branch=='R') & (df_Ar_HITRAN.quanta_m<20)].quanta_m)


df_Ar_HITRAN.loc[(df_Ar_HITRAN.quanta_branch=='R') & (df_Ar_HITRAN.quanta_m>=20) & (df_Ar_HITRAN.quanta_UL==10), 'delta_air'] = Pade_22(
    df_Ar_HITRAN[(df_Ar_HITRAN.quanta_branch=='R') & (df_Ar_HITRAN.quanta_m>=20) & (df_Ar_HITRAN.quanta_UL==10)].quanta_m, *parameters_delta_air_Pade_22)


df_Ar_HITRAN.loc[df_Ar_HITRAN.quanta_branch=='P', 'delta_air'] = const_then_slope(df_Ar_HITRAN[df_Ar_HITRAN.quanta_branch=='P'].quanta_m, *delta_parameters_Ar_CtSP)


transitions_UL = [10, 21] # [10, 21]


for prop in ['delta_air', 'gamma_air', 'n_air']:
        
    plot_fit([df_CO, df_Ar_HITRAN], ['HITRAN (Air)', 'Argon'], ['quanta_m', prop], ['m', prop], [], [], [], transitions_UL)



db.df_to_par(df_Ar_HITRAN, par_name = molecule+'_Ar', save_dir=d_database)



















