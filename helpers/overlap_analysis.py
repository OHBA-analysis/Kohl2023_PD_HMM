#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:35:53 2022

@author: okohl
"""
import numpy as np
from sklearn.metrics import jaccard_score
import random

def get_overlap(arr_in,vect_comp,jaccard_index = False):
    '''
    Calculate overlap metrics between comparison vector and input array

    Parameters
    ----------
    arr_in : np.array
        Our thresholded Gamma (states x times).
    vect_comp : np.array
        Is_burst vector.
    jaccard_index : Bool
        Do you want to calculate Jaccard index? This is quite time consuming. 
        The default is False.

    Returns
    -------
    Returns Overlap Metrics:
        Absolute Overlap,
        Overlap normalised against burst duration
        Overlap normalised against state durations
        if selected: Jaccard Index
    '''
    
    
    # Get nRow and nCol of input array
    [row,col] = arr_in.shape
        
    overlap =  np.logical_and(arr_in,vect_comp).sum(axis=1) # Get Overlap between Gamma and comp vect
    
    # calculate overlap metrics and store them
    overlap_abs = overlap
    overlap_compNorm = overlap/vect_comp.sum() # normalise against burst time in samples
    overlap_arrInNorm = overlap/arr_in.sum(axis=1) # normalise against state times in samples
    
    if jaccard_index:
        jaccard_ind = np.array([jaccard_score(arr_in[iRow],vect_comp) for iRow in range(row)])
        
        return overlap_abs, overlap_compNorm, overlap_arrInNorm, jaccard_ind
    
    else:
        return overlap_abs, overlap_compNorm, overlap_arrInNorm    


def get_overlap_nulls(arr_in, vect_comp, nPerm = 1000, jaccard_index = False, random_seed = False):
    '''
    Calculate overlap metrics between comparison vector and rolled versions of 
    input array to construct null distributions for later significance testing.

    Parameters
    ----------
    arr_in : np.array
        Our thresholded Gamma (states x times).
    vect_comp : np.array
        Is_burst vector.
    nPerm : Int
        Number of permutations. 
        The default is 1000.
    jaccard_index : Bool
        Do you want to calculate Jaccard index? This is quite time consuming. 
        The default is False.
    random_seed: int
        If you want to use this function in loop and want same shfits of array
        in each iteration give random seed an int. Otherwise randome shifts
        in each iteration
        The default is False.

    Returns
    -------
    Returns Overlap Metrics:
        Absolute Overlap,
        Overlap normalised against burst duration
        Overlap normalised against state durations
        if selected: Jaccard Index
    '''

    # Set seed if specified
    if random_seed:
        random.seed(random_seed)
        
    # Get nRow and nCol of input array
    [row,col] = arr_in.shape
    
    # Randome shifts of circular rolling
    rand = random.sample(range(col),nPerm)
    
    # Preallocation
    overlap_abs = np.empty([nPerm,row])
    overlap_compNorm = np.empty([nPerm,row])
    overlap_arrInNorm = np.empty([nPerm,row])
    jaccard_ind = overlap_compNorm = np.empty([nPerm,row])
    for ind, i in enumerate(rand):
        rand_arr = np.roll(arr_in,i,axis=1) # Roll Gammas
        overlap =  np.logical_and(rand_arr,vect_comp).sum(axis=1) # Get Overlap between Gamma and comp vect
        
        # calculate overlap metrics and store them
        overlap_abs[ind] = overlap
        overlap_compNorm[ind] = overlap/vect_comp.sum() # normalise against burst time in samples
        overlap_arrInNorm[ind] = overlap/arr_in.sum(axis=1) # normalise against state times in samples
        
        if jaccard_index:
            jaccard_ind[ind] = np.array([jaccard_score(rand_arr[iRow,:],vect_comp) for iRow in range(row)])
            
    if jaccard_index:
        return overlap_abs, overlap_compNorm, overlap_arrInNorm, jaccard_ind
    
    else:
        return overlap_abs, overlap_compNorm, overlap_arrInNorm