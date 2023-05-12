#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:56:57 2023

@author: okohl

Project Beta Power for each parcel on surface plot to show spatial distribution of
beta power during burst events. -> Figure 5
"""

import os
import numpy as np
from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting


# --- Set Dirs and Parameters ---

# HMM parameters
run = 7
ds = 250
K = 8

# Dirs
proj_dir = "/path/to/proj_dir"
indir = proj_dir + '/Data/BurstSpectra/ds' + str(ds) + '/K' + str(K) + '/run' + str(run) + '/'
output_dir = proj_dir + 'Results/Amp_Bursts/StateDescriptions/'

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "path/to/osl/parcellations/fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"


# --- Load Data ---

# Load spectra
f = np.load(indir + "singleSub/Subject1_f.npy") # freq vect same for all participants
psd = np.load(indir + "psd_all.npy")
coh = np.load(indir + "coh_all.npy")

#Average across Participants
psd = np.mean(psd,axis=0)
coh = np.mean(coh,axis=0)

# Only keep regression coefficients for the PSD -
# -> allows to show change in frome mean beta power when Bursts are present
psd = psd[0]


# --- Plot power maps of beta power for burst on and off events ---

# Grab Diagnonals in beta range + make surface Plot
power_map = power.variance_from_spectra(f, psd, frequency_range=[15,25])
power.save(
    power_map=power_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=f"{output_dir}/clusterBeta_power_.svg",
    subtract_mean=True,
    plot_kwargs={
        "cmap": "RdBu_r",
        "bg_on_data": 1,
        "darkness": .4,
        "alpha": 1}
)
plotting.close()


