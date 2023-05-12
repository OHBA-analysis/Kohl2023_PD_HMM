#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:30:03 2023

@author: okohl

Script Calculates TDE-HMM state specific Power Spectra and Coherence with
regression method.

Within-Participant GLM framework:
    Time courses of participants devided in windows. For each of these window
    state fractional occupancies and power are calculated. State Probabilities
    are regressed onto power to estimate, how changes in state fractional
    occupancies influence power.
"""

import numpy as np
from scipy import io
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

# Hmm Parameters
run = 7
ds = 250
nSub = 67
K = 8
nParc = 39

number_of_states = [8,10,12]
runs = [1,2,3,4,5,6,7,8,9,10]

for K in number_of_states:
    for run in runs:

        # Set dirs
        proj_dir = '/path/to/proj_dir'
        indir = proj_dir + '/spectra_toPython/ds'+ str(ds) + '/K' + str(K) + '/run' + str(run)
        outdir = proj_dir + '/StateSpectra/ds'+ str(ds) + '/K' + str(K) + '/run' + str(run) + '/Spectra/'
        
        # Load Data and Gamma
        psd_all = np.zeros([nSub,2,K,nParc,47])
        coh_all = np.zeros([nSub,K,nParc,nParc,47])
        for i in range(1, nSub+1):
            
            print(f"{indir}/Subject{i}_HMMout.mat")
            
            mat_file = io.loadmat(f"{indir}/Subject{i}_HMMout.mat", simplify_cells=True)
            ts = [mat_file['subj_data']]
            alp = [mat_file['subj_Gamma']]
            
            del mat_file
        
            # Calculate mode spectra
            f, psd, coh, w = spectral.regression_spectra(
                data=ts, # time series
                alpha=alp, # Viabi Path
                sampling_frequency=250,
                window_length=1000,
                frequency_range=[0, 45],
                step_size=20,
                n_sub_windows=8,
                return_weights=True,
                return_coef_int=True,
                standardize=True
            )
            
            psd_all[i-1,:,:,:,:] = psd
            coh_all[i-1,:,:,:,:] = coh
            
        # Save
        np.save(outdir + "/psd_all.npy", psd_all)
        np.save(outdir + "/coh_all.npy", coh_all)
        
        
        
        
        
        
        
