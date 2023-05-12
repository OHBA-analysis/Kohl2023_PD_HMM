#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

Script calculates network associated beta bursts (NABBs):

    Periods with overlap between occurence of a brain state and beta bursts
    from amplitude thresholding analysis are identified. Co-Occurences are
    stored in binary co-occurence vs no-co-occurence vector.
    From this binary co-occurence time course state metrics of NABBs are calculated.
    
    Importantly, NABB metrics and binary NABB on-vs.-off
    vectors are saved with this script!
    
This step is repeated for nreapeat and number_of_states to get NABB metrics
for all HMM fits. These values will be used for the NABB Robustness analyses.

"""

import numpy as np
from scipy.io import loadmat
import pickle


import sys

sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")

from burst_analysis import burst_detection, burst_time_metrics, custom_burst_metric
    

#%% Get State Burst Overlap and Metrics


# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
number_of_states = [8,10,12]
fs = 250
nrepeate = [1,2,3,4,5,6,7,8,9,10]#[1,2,3,4,5]

for K in number_of_states:
    proj_dir = '/home/okohl/Documents/HMM_PD_V07/'
    outdir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/'
    data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'
    data_outdir = proj_dir + 'Data/Burst_x_State_Metrics/ds250/K' + str(K) + '/'
    
    
    # ---- Start Loop  through outputs of different HMM runs
    for irun in nrepeate: 
        
        # --- Start Loop through Participants to save Copes and Tstats ---   
        burst_overlap = np.zeros([nSub])
        burst_FOs = np.zeros([nSub,K,1])
        burst_rates = np.zeros([nSub,K,1])
        burst_meanLTs = np.zeros([nSub,K,1])
        burst_meanITs = np.zeros([nSub,K,1])
        burst_meanAmps = np.zeros([nSub,K,1])
        
        for iSub in range(nSub):
            
            print('Loading Data for Subject' + str(iSub+1))
            
            # --- Load Data ---
            # Load Data
            file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
            data_in = loadmat(data_dir + file)
            XX_r = data_in['subj_data'][:,17] # Just Select Parcel 17 or 18
            XX_l = data_in['subj_data'][:,18]
            Gamma = data_in['subj_Gamma']
            
            # Binarise state probabilities
            Gamma_bin = np.array([Gamma[:,kk] > .75 for kk in range(K)]).astype(float)
            
            # --- Extract Beta Bursts ----
            
            # Get Burst on off vector and normalised data
            freq_range = [18, 25]
            fsample = 250
    
            is_burst_l, norm_data_l = burst_detection(XX_l, freq_range, fsample = fsample, normalise = 'zscore_no', 
                                                  threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
            
            is_burst_r, norm_data_r = burst_detection(XX_r, freq_range, fsample = fsample, normalise = 'zscore_no', 
                                                  threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
    
            
            is_burst = np.logical_or(is_burst_l,is_burst_r)
            norm_data = np.mean([np.squeeze(norm_data_l), np.squeeze(norm_data_r)],axis = 0)
            
            # How do Bursts in left and right parcel overlap ?
            burst_overlap[iSub] = sum(is_burst_l == is_burst_r)/len(is_burst_l)
            print(str(np.round(burst_overlap[iSub],2)*100) + '% Overlap between right hemisphere and left hemisphere Bursts.')
            
            # --- Combine Burst and State Vector ---
            
            burst_amps = []
            burst_lifetimes = []
            is_BurstState_out = np.zeros(Gamma_bin.shape)
            # ---- Calculate overlap between Bursts and States -----
            for iState in range(Gamma_bin.shape[0]):
                
                is_BurstState = np.logical_and(Gamma_bin[iState,:],is_burst)
                      
                # ----- Calculate Burst Metrics -----
                Mean Amplitude
                # Get Burst Time Metrics
                burst_time_dict = burst_time_metrics(is_BurstState, fsample)
                
                # Get Sinle Burst Mean Amplitudes and Lifetimes
                burst_amps = np.array(custom_burst_metric(norm_data,burst_time_dict['Starts'], 
                                                  burst_time_dict['Ends'], func = np.mean))
                
                # Get Average Measures
                burst_FOs[iSub,iState] = burst_time_dict['Burst Occupancy']
                burst_rates[iSub,iState] = burst_time_dict['Burst Rate']
                burst_meanLTs[iSub,iState] = np.nanmean(burst_time_dict['Life Times'])
                burst_meanITs[iSub,iState] = np.nanmean(burst_time_dict['Interval Times'])
                burst_meanAmps[iSub,iState] = np.nanmean(burst_amps)
                
                is_BurstState_out[iState,:] = is_BurstState
                
                # Save binary vector indicating State_x_Burst overlap // different betas
                np.save(data_outdir + '/run' + str(irun) + '/is_StateBurst/is_StateBurst_Subject' + str(iSub), is_BurstState_out) 
            
        # Prepare dat
        MetricLabels = ['Fractional Occupancy', 'Mean Lifetimes', 'Mean Interval Times','Burst Rates', 'Mean Power']
        allMetrics = [burst_FOs, burst_meanLTs, burst_meanITs, burst_rates, burst_meanAmps]     
        
        # ------ Results for further Plotting ------
        mdict = {'metrics': allMetrics, 'metric labels': MetricLabels, 'burst_overlap': burst_overlap}
        pickle.dump(mdict, open(outdir + 'run' + str(irun) + '/Burst_x_State_Metrics.mat', "wb"))
        
        # ------- Results with Glm Spectrum Beta Power as Amplitude Measure -------
        
        #  Load GLM-Spectrum Output 
        spectrum_dir = proj_dir + 'Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        beta_spectrum = loadmat(spectrum_dir + 'StateBurst_GLM_BetaPower.mat')['out'][:,:,np.newaxis]

        # Overwrite Beta Ampl Envelope Power Calculated Above with GLM Spectrum results 
        allMetrics[4] = beta_spectrum

        # Save Data
        mdict = {'metrics': allMetrics, 'metric labels': MetricLabels}
        pickle.dump(mdict, open(outdir + 'run' + str(irun) + '/Burst_x_State_Metrics_GLM_Pow.mat', "wb"))
