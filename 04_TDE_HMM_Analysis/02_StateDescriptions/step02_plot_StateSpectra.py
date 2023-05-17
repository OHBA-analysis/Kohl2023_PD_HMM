#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:30:03 2023

@author: okohl

Plot HMM resting-state PSDs, power maps, and coherence maps based on outputs
from step01.

Note: When you run the script for the first time you wont have the sorting vector
because it will be calculated based on the outputs from this scripts in step 04
in this folder. In this case comment the line loading the sorting vector and the
brackets ([:,state_sorting]) behind the loading of the psds, cohs, and fos.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

import sys
sys.path.append("/path/to/helpers/")
from plotting import get_colors, tsplot

# HMM parameters
ds = 250
run = 7
K = 8

# Set Dirs
proj_dir = "/path/to/proj_dir/"
indir = proj_dir + "Data/StateSpectra/ds" + str(ds) + "/K" + str(K) + "/run" + str(run) + "/Spectra/"
fo_dir =  proj_dir + 'Data/StateMetrics/ds' + str(ds) + '/K' + str(K) + '/run' + str(run) + '/'
output_dir = proj_dir + "Results/State_Descriptions/ds" + str(ds) + "/K" + str(K) + "/run" + str(run) + "/Spectra/"

# Make output directory
os.makedirs(output_dir, exist_ok=True)

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "/path/to/osl/parcellations/fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"

# Load State Sorting vector
sort_states = np.load(proj_dir + '/Data/sortingVector/ClusterPowSorting_mt.npy')

# Load spectra + sort states
f = np.load(indir + "f.npy")
psd = np.load(indir + "psd.npy")[:,sort_states]
coh = np.load(indir + "coh.npy")[:,sort_states]
fo = loadmat(fo_dir + 'fractional_occupancy.mat')['out'][:,sort_states]

# Subtract Mean Across States
mean_psd = np.empty([psd.shape[0],psd.shape[2],psd.shape[3]])
psd_nomean = np.empty([psd.shape[0],K,psd.shape[2],psd.shape[3]])
for iSub in range(fo.shape[0]):
    mean_psd[iSub] = np.average(psd[iSub], axis=0, weights=fo[iSub])
    psd_nomean[iSub] = psd[iSub] - mean_psd[iSub]
         
    
# --- Plot State PSDs ---

# Get Colors for Plotting
col = get_colors()['PDvsHC_bin'][1]

# Average across motor parcels
psd_in = np.mean(psd[:,:,[17,18]],axis=2)
psd_in = psd_in.transpose([1,0,2]) # Bring into format for plotting
mean_psd = np.mean(mean_psd[:,[17,18]],axis=1)

# Set Upper Frquency limit for plot
f_in = f[f <= 30]
mean_psd_in = mean_psd[:,f <= 30]
psd_in = psd_in[:,:,f <= 30]

# Combine mean and state specific changes - can be also calculate by simply averaging psd loaded at beginning of script
psd_in = psd_in + mean_psd_in

for iState in range(psd_in.shape[0]):
    
    # Start Plotting
    fig, ax = plt.subplots()
    tsplot(ax, psd_in[iState], mean_psd_in, time = f_in, color_data = col, color_mean='grey')
    
    #Save Figure
    plt.savefig(output_dir + '/motor_PowSpec_State' + str(iState) + '_wMean.svg',
                transparent = True,bbox_inches="tight",format='svg')


# --- Average across Participants ---
mean_psd = np.mean(mean_psd,axis=0)
psd_nomean = np.mean(psd_nomean,axis=0)
psd = np.mean(psd,axis=0)
coh = np.mean(coh,axis=0)


# --- Plot power maps ---
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


# --- Plot coherence maps ---
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






