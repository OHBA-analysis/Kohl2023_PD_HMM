#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:59:32 2022

@author: okohl

Script Extracts mean power across significant beta cluster from State specific Spectra
of Network Associated Beta Bursts.

Beta Power will be input for NABB metric Group Contrast.
"""

import numpy as np
from scipy.io import loadmat,savemat

number_of_states = [8,10,12]
runs = [1,2,3,4,5,6,7,8,9,10]
fs = 250

for K in number_of_states:
    for run in runs:
        
        # Set Dirs
        proj_dir = '/home/okohl/Documents/HMM_PD_V07/Data'
        pow_dir = proj_dir + '/StateBurst_Spectra/ds'+ str(fs) + '/K' + str(K) + '/run' + str(run) + '/Multitaper/'
        metric_out = proj_dir + '/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(run) + '/'
        fo_dir =  proj_dir + '/StateMetrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(run) + '/'
        
        # Load spectra
        f = np.load(pow_dir + "f.npy")
        psd = np.load(pow_dir + "psd.npy")
        w = np.load(pow_dir + 'w.npy')
        fo = loadmat(fo_dir + 'fractional_occupancy.mat')['out']

        for iSub in range(fo.shape[0]):
            psd -= np.average(psd, axis=1, weights=fo[iSub])[:,np.newaxis,...]
        
        # Average Across Beta Range
        beta = np.logical_and(f >= 18, f<=25)
        psd = np.mean(psd[:,:,:,beta],axis=3)
        
        # Average across Motor Channels
        psd = np.mean(psd[:,:,17:19],axis=2)
        
        # Save Data
        mdict = {'out': psd}
        savemat(metric_out + 'StateBurst_BetaPower.mat',mdict)