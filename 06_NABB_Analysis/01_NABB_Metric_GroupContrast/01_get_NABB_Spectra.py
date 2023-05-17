#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:17:14 2023

@author: okohl

Calculates Network Informed Beta Burst Spectra using multitaper method.
Outputs will be used to calculate NABB average beta power which will be an 
input for Figure 6.

Loop across different number of states and runs for calculation of Robustness
of group contrasts and correlations. -> SI7 & SI8

"""

import numpy as np
from scipy import io
from osl_dynamics.analysis import spectral

# Hmm Parameters
runs = [1,2,3,4,5,6,7,8,9,10]
ds = 250
nSub = 67
number_of_states = [12]
nParc = 39

for K in number_of_states:
    for run in runs:

        # Set dirs
        proj_dir = 'path/to/proj_dir/Data/'
        indir = proj_dir + 'spectra_toPython/ds'+ str(ds) + '/K' + str(K) + '/run' + str(run)
        indir_StateBurst = proj_dir + 'Burst_x_State_Metrics/ds250/K' + str(K) + '/run' + str(run) + '/is_StateBurst/'
        outdir = proj_dir + '/StateBurst_Spectra/ds'+ str(ds) + '/K' + str(K) + '/run' + str(run) + '/ChetsSpectra'
        
        # Load Data and Gamma
        data = []
        alpha = []
        for i in range(1, nSub+1):
            print(f"{indir}/Subject{i}_HMMout.mat")
            
            mat = io.loadmat(f"{indir}/Subject{i}_HMMout.mat", simplify_cells=True)
            data.append(mat["subj_data"])
            alpha.append(mat["subj_Gamma"])

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
        np.save(outdir + "/psd.npy", psd)
        np.save(outdir + "/coh.npy", coh)
        np.save(outdir + "/f.npy", f)
        np.save(outdir + "/w.npy", w)