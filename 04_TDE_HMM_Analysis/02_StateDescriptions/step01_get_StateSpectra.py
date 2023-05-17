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
        
        
        
        
        
        
        
