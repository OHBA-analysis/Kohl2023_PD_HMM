#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:00:39 2022

@author: okohl

Script loads Fractional Occupancy of Sensorimotor State and Static Power per Participant.
18 to 25Hz average power is calculated for each participant and calculated with
Fractional Occupancy valyes.

Saved Data is input for Figure 3.
"""


import numpy as np
import pandas as pd
import scipy.io as io
from scipy.stats import pearsonr
import pickle

# --- Set Parameters and Dirs ---

# Parameter
nSub = 67
K = 8
fsample = 250
irun = 7
sub_split = 36  # 36 HCs and 30 PD patients

# Set Dirs
proj_dir = '/path/to/proj_dir/'
in_dir = proj_dir +  'Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

outdir_dat = proj_dir + 'Data/StateMetrics_x_Pow/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

outdir_plot = proj_dir + 'Results/StateMetrics_x_Pow/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/allSub/'

pow_dir = proj_dir + '/Data/staticSpectra/'


# --- load Covariates ----
behav_in = '/path/to/behavioral_data/'
df = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')

# Remove Rows with nan and store indices of good Rows
df_in = df[['Group', 'Handedness', 'Gender', 'Age', 'Education']] # UPDRS motor just added to identify Pd patients
in_ind = np.prod(df_in.notna().values, axis=1)
in_ind = np.ma.make_mask(in_ind)
df_in = df_in[in_ind]


# --- Load Fractional Occupancies ---
labels_metric = 'fractional_occupancy'
labels = ['State1', 'State2', 'State3', 'State4', 'State5', 'State6',
          'State7', 'State8']

in_data = io.loadmat(in_dir + labels_metric + ".mat")['out']
df_states_in = pd.DataFrame(in_data, columns=labels)
df_states = df_states_in['State8'][in_ind]


# --- Load Copes/Power Spectra ---
Level1 = pickle.load(open( pow_dir + "zscore_pow_all.dat", "rb" ) )

# lose low freqs - 1/f can dominate stats
freqs = Level1['Freqs']
keeps = np.where(np.logical_and(freqs>=18,freqs <= 25))[0]
freqs = freqs[keeps]

# Grab Copes of interest
copes = Level1['Pow'][in_ind,:,:][:,keeps,:]
mean_copes = np.mean(copes,axis = 1)

# Average across all parcels
mean_copes = np.mean(mean_copes[:,17:19],axis = 1)

# --- Correlation between Power and FO ----
[r,p] = pearsonr(mean_copes,df_states.values)



# --- Perpare Export of Data ---

# Find PDs
PDs = df_in['Group'] == 1
PDs = PDs.values

# Df for plotting Elsewhere
dat_plot = {'Fractional Occupancy': df_states.values,
            'Beta Power' : mean_copes}
dat_plot = pd.DataFrame(dat_plot)


# --- Save Data for Further Plotting elsewhere ---
mdict = {'dat_plot': dat_plot, 'PDs':PDs, 'r':r,'p':p}
pickle.dump(mdict, open(proj_dir + '/Data/V2_Plots/FO_Power_correl.p', "wb"))
