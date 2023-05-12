#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:30:03 2023

@author: okohl

Plot HMM resting-state PSDs, power maps, and coherence maps based on outputs
from step01.
"""

import os
import numpy as np
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

# HMM parameters
ds = 250
run = 7
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


# Load spectra
f = np.load(indir + "singleSub/Subject1_f.npy")
psd = np.load(indir + "psd_all.npy")
coh = np.load(indir + "coh_all.npy")

#Average across Participants
psd = np.mean(psd,axis=0)
coh = np.mean(coh,axis=0)

# Only keep regression coefficients for the PSD -> power changes from mean when state occurres
psd = psd[0]

# Plot mode PSDs
p = np.mean(psd, axis=1)  # mean over channels
e = np.std(psd, axis=1) / np.sqrt(psd.shape[1])
for i in range(p.shape[0]):
    fig, ax = plotting.plot_line(
        [f],
        [p[i]],
        labels=[f"State {i + 1}"],
        errors=[[p[i] - e[i]], [p[i] + e[i]]],
        x_range=[3, 30],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        fig_kwargs={"figsize": (7, 5)},
    )
    ax.legend(handlelength=0)
    plotting.save(fig, f"{output_dir}/psd{i}.svg")
plotting.close()


# Plot power maps
power_map = power.variance_from_spectra(f, psd, frequency_range = [2,30])
power.save(
    power_map=power_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=f"{output_dir}/power_.png",
    subtract_mean=True,
    plot_kwargs={
        "cmap" : "RdBu_r",
        "bg_on_data" : 1,
        "darkness" : .4,
        "alpha" : 1}
)
plotting.close()


# Plot coherence maps
conn_map = connectivity.mean_coherence_from_spectra(f, coh, frequency_range=[2, 20]) #2:20
conn_map = connectivity.threshold(conn_map, percentile=98, subtract_mean=True)
connectivity.save(
    connectivity_map=conn_map,
    filename=f"{output_dir}/conn_.svg",
    parcellation_file=parcellation_file,
    plot_kwargs={
        "edge_cmap":"red_transparent_full_alpha_range",
        "display_mode":"lyrz"}
)






