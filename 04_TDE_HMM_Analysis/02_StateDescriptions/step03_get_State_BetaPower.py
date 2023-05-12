#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:59:32 2022

@author: okohl

Script Extracts mean power across significant beta cluster from GLM-Spectrum
calculated State Power Spectra.
"""

import numpy as np
from scipy.io import savemat

fs = 250
number_of_states = [8,10,12]
runs = [1,2,3,4,5,6,7,8,9,10]

for K in number_of_states:
    for run in runs:

        # Set Dirs
        proj_dir = '/home/okohl/Documents/HMM_PD_V07/Data'
        pow_dir = proj_dir + '/StateSpectra/ds'+ str(fs) + '/K' + str(K) + '/run' + str(run) + '/Spectra/'
        metric_out = proj_dir + '/StateMetrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(run) + '/'
        
        # Load spectra
        f = np.load(pow_dir + "singleSub/Subject1_f.npy")
        psd = np.load(pow_dir + "psd_all.npy")
        
        # Only keep regression coefficients for the PSD - ignore intercept
        psd = psd[:,0]
        
        # Average Across Beta Range
        beta = np.logical_and(f >= 18, f<=25)
        psd = np.mean(psd[:,:,:,beta],axis=3)
        
        # Average across Motor Channels
        psd = np.mean(psd[:,:,17:19],axis=2)
        
        # Save Data
        mdict = {'out': psd}
        savemat(metric_out + 'State_GLM_BetaPower.mat',mdict)