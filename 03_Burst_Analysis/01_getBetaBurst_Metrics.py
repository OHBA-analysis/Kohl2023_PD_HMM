#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

Script Calculates Burst Metrics that are fed into group contrast in Fig2 in
Manuscript.

Steps:
    1. Load Data
    2. Burst analysis for left and right motor cortex
        a. Filter Power to 18 to 25Hz range. (Sig Cluster in Static Analysis)
        b. Amplitude Envelopes calculated with Hilbert Transform.
        c. Amplitude Envelopes thresholded at the 75th Percentile.
        d. Just keep instances with duration longer than 1 cycle of lowest beta frequency        
    3. Combining of on vs off burst vectors of both motor parcels.
    4. Calculation of Burst Metrics
    5. Is burst vector is stored for calculation of power of bursts with GLM-Periodogram.
    6. Burst Metrics are saved.
"""

import numpy as np
from scipy.io import loadmat
import pickle

import sys
sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")
from burst_analysis import burst_detection, burst_time_metrics, custom_burst_metric


# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]#[1,2,3,4,5]

proj_dir = '/home/okohl/Documents/HMM_PD_V07/'
out_dir_plot = proj_dir + 'Results/Burst_Overlap/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'


# ---- Start Loop  through outputs of different HMM runs
for irun in nrepeate: 
    
    # --- Start Loop through Participants to save Copes and Tstats ---
    burst_amps = []
    burst_lifetimes = []
    burst_FOs = np.zeros([nSub,1])
    burst_rates = np.zeros([nSub,1])
    burst_meanLTs = np.zeros([nSub,1])
    overlap = np.zeros([nSub,K])
    overlap_shuff = np.zeros([nSub,K])
    
    for iSub in range(nSub):
        
        print('Loading Data for Subject' + str(iSub+1))
        
        # --- Load Data ---
        # Load Data
        file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
        data_in = loadmat(data_dir + file)
        XX_r = data_in['subj_data'][:,17] # Just Select Parcel 17 or 18
        XX_l = data_in['subj_data'][:,18]
         
        # Extract Beta Bursts
        
        # Get Burst on off vector and normalised data
        freq_range = [18, 25]
        fsample = 250

        is_burst_l, _ = burst_detection(XX_l, freq_range, fsample = fsample, normalise = 'none', 
                                              threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
        
        is_burst_r, _ = burst_detection(XX_r, freq_range, fsample = fsample, normalise = 'none', 
                                              threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)

             
        is_burst = np.logical_or(is_burst_l,is_burst_r)
        
        # ----- Calculate Burst Metrics -----
        
        # Get Burst Time Metrics
        burst_time_dict = burst_time_metrics(is_burst, fsample)
        
        # Get Sinle Burst Lifetimes      
        burst_lifetimes.append(burst_time_dict['Life Times'])
        
        # Get Average Measures
        burst_FOs[iSub,:] = burst_time_dict['Burst Occupancy']
        burst_rates[iSub,:] = burst_time_dict['Burst Rate']
        burst_meanLTs[iSub,:] = np.nanmean(burst_time_dict['Life Times'])
        
        # --- Save is Burst / Separate File Per Participant
        np.save(data_dir + '/isBurst/isBurst_Subject' + str(iSub), is_burst)
 
        
#%% Save Burst Metrics with Amp Envelope as Beta Power measure

outdir = '/path/to/BurstMetrics/ds250/K8/run7/'

# Prepare dat
MetricLabels = ['Fractional Occupancy', 'Burst Rates','Mean Lifetimes']
allMetrics = [burst_FOs,burst_rates, burst_meanLTs]

# ------ Results for further Plotting ------
mdict = {'metrics': allMetrics, 'labels': MetricLabels}
pickle.dump(mdict, open(outdir + 'BurstMetrics.mat', "wb"))
