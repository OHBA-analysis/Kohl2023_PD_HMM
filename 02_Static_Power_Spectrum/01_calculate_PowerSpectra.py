#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:30:37 2022

@author: okohl

Script calulating Power Spectrum for each Participant with Welch's Method
implemented in Sails framework (Quinn et al., 2022).

1) HMM-input data for each participant is imported (times x parcel matrix))
2) z-Scores of data are calculated
3) 1-Level Welch's Method GLM only modeling the mean is applied (= Welch's Method)
4) Data of all participant is concatenated for storing.

"""

import numpy as np
from scipy.stats import zscore
from scipy.io import loadmat
import pickle
import sails
from scipy.signal import welch

# --- Set Parameters and Dirs ----
proj_dir = '/path/to/proj_dir/'
out_dir = proj_dir + 'Data/staticSpectra/'
data_dir = proj_dir + 'Data/spectra_toPython/ds250/K8/run7/'


# --- Run Loop loading & zscoring data + calculating glm_periodogram ---
nSub = 67
fs = 250
copes_all = np.empty([nSub,89,39])
for iSub in range(nSub):
    
    print('Running Sub' + str(iSub + 1))
    
    # Load Data
    file = 'Subject' + str(iSub + 1) + '_HMMout.mat'    
    data = loadmat(data_dir + file)['subj_data']
    
    # z-Scoring
    z_data = zscore(data,axis=0) 
    
    #Calculate Subject Power Spectrum with GLM-Periodogram only modeling the mean; is equal to Welch's Method
    freq_vect, copes, _, _ = sails.stft.glm_periodogram(z_data, 
                                                        axis=0,
                                                        fit_constant=True,
                                                        nperseg=fs*2,
                                                        noverlap=fs,
                                                        fmin=1, fmax=45,
                                                        fs=fs,
                                                        mode='magnitude',
                                                        fit_method='glmtools')
    
    # Alternatively, use scipys' Welchs Method
    #freq_vect, copes = welch(z_data, fs, nperseg=fs*2, noverlap=fs, axis=0)  # Works with single or multi-channel data
    
    # Store Matrix of all Sub in Matrix
    copes_all[iSub,:,:] = copes
   
# --- Store Data ---
pow_out = {'Pow': copes_all, 'Freqs': freq_vect}
pickle.dump( pow_out, open( out_dir + "zscore_pow_all.dat", "wb" ) )