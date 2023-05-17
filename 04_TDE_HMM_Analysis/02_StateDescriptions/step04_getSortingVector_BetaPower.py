#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:15:03 2022

@author: okohl

Script extracts vector to sort states according to their 18 to 25Hz beta power.

"""

import numpy as np
from scipy.io import loadmat

# --- Set Dirs and Parameters ---

# HMM parameters
run = 7
ds = 250
K = 8

# Set Dirs
proj_dir = '/home/okohl/Documents/HMM_PD_V07/Data'
pow_dir = proj_dir + '/StateSpectra/ds'+ str(ds) + '/K' + str(K) + '/run' + str(run) + '/Spectra/'
fo_dir =  proj_dir + 'Data/StateMetrics/ds' + str(ds) + '/K' + str(K) + '/run' + str(run) + '/'

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "/path/to/osl/parcellations/fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"

# --- Load Spectra and extracte 18 to 25Hz average Power ---
f = np.load(pow_dir + "f.npy")
psd = np.load(pow_dir + "psd.npy")
coh = np.load(pow_dir + "coh.npy")
fo = loadmat(fo_dir + 'fractional_occupancy.mat')['out']
        
# Subtract Mean Across States
for iSub in range(fo.shape[0]):
        psd -= np.average(psd, axis=1, weights=fo[iSub])[:,np.newaxis,...]

#Average across Participants
psd = np.mean(psd,axis=0)
coh = np.mean(coh,axis=0)

# Get motor cortical parcel average
p = np.mean(psd[:,17:18], axis=1)  # mean over channels

# Get Beta Power Average
beta = np.logical_and(f >= 18, f <= 25)
p = np.mean(p[:,beta],axis=1)

# --- Get State Sorting based on 18 to 25Hz average motor cortical power ---
State_sorting = np.argsort(-p)

# --- Save Sorting Vector ---
np.save(proj_dir + '/sortingVector/ClusterPowSorting.npy',State_sorting)

