#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

Script compares burst overlap with HMM state overlap.

Steps:
    1. Load Data and Burst on vs off time courses
    3. GLM calculated per Participants
        a. window time course and calculate power spectrum for each window
        b. apply same windowing to binarised Gamma time course
        c. calculate GLMs predicting Power Spectra across windows based on
           bin Gamma time courses of all States. TCs are treated as confounds,
           thus are modeling the mean of the power spectral data as well!
"""

import numpy as np
from scipy.io import loadmat
from scipy import io
from osl_dynamics.analysis import spectral

# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]
nParc = 39

proj_dir = '/path/to/proj_dir/'
outdir = proj_dir + '/Data/BurstSpectra//ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'


# Preallocate Arrays
psd_all = np.zeros([nSub,2,2,nParc,47])
coh_all = np.zeros([nSub,2,nParc,nParc,47])

# ---- Start Loop  through outputs of different HMM runs
for irun in nrepeate: 
    
    # --- Start Loop through Participants to save Copes and Tstats ---
    
    for iSub in range(nSub):
        
        print('Loading Data for Subject' + str(iSub+1))
        
        # --- Load Data ---
        # Load Data
        file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
        data_in = loadmat(data_dir + file)
        data = data_in['subj_data']
        
        isBurst = np.load(data_dir + '/run' + str(irun) + '/isBurst/isBurst_Subject' + str(iSub) + '.npy')    


        # Calculate burst spectra
        f, psd, coh, w = spectral.regression_spectra(
            data=data, # time series
            alpha=isBurst[:,np.newaxis], # Viabi Path
            sampling_frequency=250,
            window_length=1000,
            frequency_range=[0, 45],
            step_size=20,
            n_sub_windows=8,
            return_weights=True,
            return_coef_int=True,
            standardize=True
        )
        
        psd_all[iSub-1,:,:,:,:] = psd
        coh_all[iSub-1,:,:,:,:] = coh
   
        
    # Save Burst on and burst off related Power and Coherence
    np.save(outdir + "/psd_all.npy", psd_all)
    np.save(outdir + "/coh_all.npy", coh_all)
    
    
