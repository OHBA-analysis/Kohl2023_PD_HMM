#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

Script compares burst overlap with HMM state overlap.

Steps:
    1. Load Data
    2. Gamma time course is binarised
    3. isBurst vector loaded
    4. Overlap between isBurst and Gammas and overlap is calculated
    5. Overlap between isBurst and shifted Gammas and overlap is calculated
    6. Save Overlap TCs and Overlap Metric
    
    
Overlap is calculated as the sum of co-occurrences devided by the sum of burst
occurences. This means the overlap will be expressed relative to burst occurences.
"In XX% of a time a burst is present whole-brain network X co-occures".


Overlap calculated between is Burst Vector and rolled versions of HMM-State
on-vs-of array to construct null distributions for later significance testing.
Procedure is repeated for 1000 roles with a randome number rolling number.
Rolling numbers of 1000 roles are the same for each particpant.
"""

import numpy as np
from scipy.io import loadmat
import pickle

import sys
sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")
from overlap_analysis import get_overlap, get_overlap_nulls
    


# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]#[1,2,3,4,5]
nPerm = 1000

proj_dir = '/home/okohl/Documents/HMM_PD_V07/'
out_dir_plot = proj_dir + 'Results/Burst_Overlap/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
outdir_dat = proj_dir + 'Data/Overlap/ds' + str(fs) + '/K' + str(K) +  '/run' + str(nrepeate[0]) + '/'
data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'
tc_outdir = proj_dir + 'Data/Burst_x_State_Metrics/ds250/K' + str(K) + '/'

# ---- Start Loop  through outputs of different HMM runs
for irun in nrepeate: 
    
    # --- Start Loop through Participants to save Copes and Tstats ---
    overlap_abs = np.zeros([nSub,K])
    overlap_BurstNorm = np.zeros([nSub,K])
    overlap_StatesNorm = np.zeros([nSub,K])
    jaccard = np.zeros([nSub,K])
    
    null_overlap_abs = np.zeros([nSub,nPerm,K])
    null_overlap_BurstNorm = np.zeros([nSub,nPerm,K])
    null_overlap_StatesNorm = np.zeros([nSub,nPerm,K])
    null_jaccard = np.zeros([nSub,nPerm,K])
    for iSub in range(nSub):
        
        print('Loading Data for Subject' + str(iSub+1))
        
        # --- Load Data ---
        # Load HMM Data Data
        file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
        Gamma = loadmat(data_dir + file)['subj_Gamma']
        
        # Binarise state probabilities
        Gamma_bin = np.array([Gamma[:,kk] > .75 for kk in range(K)]).astype(float)
        
        # Load Burst On Off Set Data
        is_burst = np.load(data_dir + '/run' + str(irun) + '/isBurst/isBurst_Subject' + str(iSub) + '.npy')
        
        # Get Overlap Metrics
        [_, overlap_BurstNorm[iSub], _] = get_overlap(
            Gamma_bin,is_burst, jaccard_index = False)
        
        # Get Overlap Metrics Nulls
        [_, null_overlap_BurstNorm[iSub], _] = get_overlap_nulls(
            Gamma_bin, is_burst, nPerm = nPerm, jaccard_index = False, random_seed = 1)
        
        # Get State x Burst overlap time courses
        is_BurstState_out = np.logical_and(Gamma_bin,is_burst)
      
    # --- Save Data   
      
    # Save Overlap Metrics
    mdict = {'Overlap_BurstNorm': overlap_BurstNorm, 
            'Overlap_BurstNorm_Nulls': null_overlap_BurstNorm, }
    pickle.dump( mdict, open( outdir_dat + "Overlap_Metrics.mat", "wb" ) )

    # Save State x Burst overlap time course:
    np.save(tc_outdir + '/run' + str(irun) + '/is_StateBurst/is_StateBurst_Subject' + str(iSub), is_BurstState_out) 
    
    
