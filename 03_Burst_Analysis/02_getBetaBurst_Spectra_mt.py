#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:00:04 2023

@author: okohl
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy import io
from osl_dynamics.analysis import spectral

# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7] 
nParc = 39


# ---- Start Loop  through outputs of different HMM runs
for irun in nrepeate: 
    
    # Set Dirs
    proj_dir = '/home/okohl/Documents/HMM_PD_V08/'
    outdir = proj_dir + '/Data/BurstSpectra/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
    
    indir = '/ohba/pi/knobre/okohl/PD/HMM-Analysis/HMM_PD_V07/Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'


    # Load Data and Gamma
    data = []
    alpha = []
    for i in range(1, nSub+1):
        print(f"{indir}/Subject{i}_HMMout.mat")
    
        mat = io.loadmat(f"{indir}/Subject{i}_HMMout.mat", simplify_cells=True)
        isBurst = np.load(indir + 'isBurst/isBurst_Subject' + str(i-1) + '.npy')
        isBurst = np.vstack([isBurst,~isBurst]).T
        data.append(mat["subj_data"])
        alpha.append(isBurst)
       

    # Run multitaper
    f, psd, coh, w = spectral.multitaper_spectra(
        data=data,
        alpha=alpha,
        sampling_frequency=250,
        time_half_bandwidth=4,
        n_tapers=7,
        frequency_range=[1, 45],
        return_weights=True,
        n_jobs=16,)    

    
    # Save Data   
    np.save(outdir + "/psd_all.npy", psd)
    np.save(outdir + "/coh_all.npy", coh)
    np.save(outdir + "/f.npy", f)
    np.save(outdir + "/w.npy", w)
    
    
    # --- Get Burst Fractional Occupacies - Important for subtracting the mean later ---
    fo = [np.mean(alpha[iSub],axis=0) for iSub in range(len(alpha))]
    np.save(outdir + "/fo.npy", fo)

