#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:17:14 2023

@author: okohl

Calculates Network Informed Beta Burst Spectra using regression method.
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
        psd_all = np.zeros([nSub,2,K,nParc,47])
        coh_all = np.zeros([nSub,K,nParc,nParc,47])
        for i in range(1, nSub+1):
            print(f"{indir}/Subject{i}_HMMout.mat")
            
            mat_file = io.loadmat(f"{indir}/Subject{i}_HMMout.mat", simplify_cells=True)
            ts = [mat_file['subj_data']]
            alp = [np.load(indir_StateBurst + 'is_StateBurst_Subject' + str(i-1) + '.npy').T] # binary vectors indicating StateBurst co occurences
            
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
            if i == 0:
                np.save(outdir + "/singleSub/Subject" + str(i) + "_f.npy", f)
                np.save(outdir + "/singleSub/Subject" + str(i) + "_psd.npy", psd)
                np.save(outdir + "/singleSub/Subject" + str(i) + "_coh.npy", coh)
            
            
        np.save(outdir + "/psd_all.npy", psd_all)
        np.save(outdir + "/coh_all.npy", coh_all)
