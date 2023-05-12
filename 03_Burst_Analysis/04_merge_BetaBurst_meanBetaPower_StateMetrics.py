#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:59:32 2022

@author: okohl

Script Extracts mean power across significant beta cluster from GLM-Spectrum
calculated Burst Power Spectra.

Beta power is merged with other State Metrics for calculation of Burst Metric
Group Contrast in step05.
"""

import numpy as np
from scipy.io import savemat
import pickle

# HMM Parameters
K = 8
run = 7
fs = 250

# Set Dirs
proj_dir = '/home/okohl/Documents/HMM_PD_V07/Data'
pow_dir = proj_dir + '/BurstSpectra/ds'+ str(fs) + '/K' + str(K) + '/run' + str(run)
metric_out = proj_dir + '/BurstMetrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(run) + '/'


#%% Get mean Power in significant beta cluster range 

# Load spectra
f = np.load(pow_dir + "/singleSub/Subject1_f.npy")
psd = np.load(pow_dir + "/psd_all.npy")

# Only keep regression coefficients for the PSD - ignore intercept
psd = psd[:,0]

# Average Across Beta Range
beta = np.logical_and(f >= 18, f<=25)
psd = np.mean(psd[:,:,:,beta],axis=3)

# Average across Motor Channels
psd = np.mean(psd[:,:,17:19],axis=2)

# Save Data
mdict = {'out': psd}
savemat(metric_out + 'Burst_GLM_BetaPower.mat',mdict)



#%% Merge Beta Power and State Metrics 

# Load State Metrics
outdir = '/path/to/BurstMetrics/ds250/K8/run7/'
mdict = pickle.load(open(outdir + 'BurstMetrics.mat', "rb"))

metrics = mdict['metrics']
labels = mdict['Metric_Labels']

# Merge Data
MetricLabels = [labels, 'Mean Power']
allMetrics = [metrics, psd]

# ------ Results for further Plotting ------
outdir = proj_dir + 'Data/BurstMetrics/ds250/K8/run7/'
mdict = {'metrics': allMetrics, 'labels': MetricLabels}
pickle.dump(mdict, open(outdir + 'BurstMetrics_GLMBeta.mat', "wb"))