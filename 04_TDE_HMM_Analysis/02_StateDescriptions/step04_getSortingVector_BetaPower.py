#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:15:03 2022

@author: okohl

Script extracts vector to sort states according to their 18 to 25Hz beta power.

"""

import os
import numpy as np


# --- Set Dirs and Parameters ---

# HMM parameters
run = 7
ds = 250
K = 8

# Set Dirs
proj_dir = "/path/to/proj_dir/"
indir = proj_dir + "Data/StateSpectra/ds" + str(ds) + "/K" + str(K) + "/run" + str(run) + "/Spectra/"
output_dir = proj_dir + "Results/State_Descriptions/ds" + str(ds) + "/K" + str(K) + "/run" + str(run) + "/Spectra/"

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "/path/to/osl/parcellations/fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"

# --- Load Spectra and extracte 18 to 25Hz average Power ---

# Load spectra
f = np.load(indir + "singleSub/Subject1_f.npy")
psd = np.load(indir + "psd_all.npy")
coh = np.load(indir + "coh_all.npy")

#Average across Participants
psd = np.mean(psd,axis=0)
coh = np.mean(coh,axis=0)

# Only keep regression coefficients for the PSD
psd = psd[0]

# Get motor cortical parcel average
p = np.mean(psd[:,17:18], axis=1)  # mean over channels

# Get Beta Power Average
beta = np.logical_and(f >= 18, f <= 25)
p = np.mean(p[:,beta],axis=1)

# --- Get State Sorting based on 18 to 25Hz average motor cortical power ---
State_sorting = np.argsort(-p)

# --- Save Sorting Vector ---
np.save('/home/okohl/Documents/HMM_PD_V07/Data/sortingVector/ClusterPowSorting.npy',State_sorting)

