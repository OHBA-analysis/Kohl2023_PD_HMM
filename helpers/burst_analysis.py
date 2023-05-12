#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:32:06 2022

@author: okohl
"""

# --- Burst Analysis Functions ----

import numpy as np
import random
import mne
from scipy.stats import zscore
from scipy import signal
#import neurodesp

def shuffle_state_TC(Gamma):
            
    # Find last element of each state visit
    v = np.argmax(Gamma,axis = 1)
    v_pad = np.pad(v,(0,1),mode = 'constant', constant_values = 20)
    diff_v = np.diff(v_pad)
    diff_v = diff_v != 0
    
    # Loop through state visits and save data in list
    v_list = []
    for visit in range(sum(diff_v)):
        visit_end = np.nonzero(diff_v)[0][0]
        v_list.append(v[:visit_end+1]) # Python indexing
        
        diff_v = diff_v[visit_end+1:]
        v = v[visit_end+1:]
    
    # Shuffle List to creat randome state TC
    random.shuffle(v_list)
    Gamma_shuff = np.concatenate(v_list)

    Gamma_bin_shuff = np.zeros([Gamma.shape[1],Gamma.shape[0]])
    for iState in range(Gamma.shape[1]):
        Gamma_bin_shuff[iState,:] = (Gamma_shuff == iState)[np.newaxis,:]

    return Gamma_bin_shuff


def threshold_data(norm_data, threshold_dict = {'Method': 'Percentile', 'threshold': 75}):    
    if threshold_dict['Method'] == 'Percentile':
        threshold = threshold_dict['threshold']
        is_burst = norm_data >= np.percentile(norm_data, q=threshold, axis = 1)
    elif threshold_dict['Method'] == 'Custom':
        threshold = threshold_dict['threshold']
        is_burst = norm_data >= threshold
    return is_burst


def _rmv_short_periods(sig, n_samples):
    """Remove periods that are equal to 1 for less than n_samples.
    
        !!!!! 
        This Function is borrowed from 
        https://neurodsp-tools.github.io/neurodsp/index.html
        
        Make Sure to Reference:
        https://doi.org/10.21105/joss.01272 
        !!!!
            
    """

    if np.sum(sig) == 0:
        return sig

    osc_changes = np.diff(1 * sig)
    osc_starts = np.where(osc_changes == 1)[0]
    osc_ends = np.where(osc_changes == -1)[0]

    if len(osc_starts) == 0:
        osc_starts = [0]
    if len(osc_ends) == 0:
        osc_ends = [len(osc_changes)]

    if osc_ends[0] < osc_starts[0]:
        osc_starts = np.insert(osc_starts, 0, 0)
    if osc_ends[-1] < osc_starts[-1]:
        osc_ends = np.append(osc_ends, len(osc_changes))

    osc_length = osc_ends - osc_starts
    osc_starts_long = osc_starts[osc_length >= n_samples]
    osc_ends_long = osc_ends[osc_length >= n_samples]

    is_osc = np.zeros(len(sig))
    for ind in range(len(osc_starts_long)):
        is_osc[osc_starts_long[ind]:osc_ends_long[ind]] = 1

    return is_osc


def burst_detection(data, freq_range, fsample, normalise = 'median', 
                    threshold_dict = {'Method': 'Percentile', 'threshold': 75},
                    min_n_cycles = 1):
    '''
    Detectes Bursts in data.
    The following Steps are applied:
        1) Filter Data to freq_range.
        2) Calculate amplitude of TC with Hilbert Transform.
        3) Normalise data by subtracting the median/mean.
        4) Threshold data with threshold percentile.
        5) Just keep bursts that are longer than min_n_cycles.
            This step uses code from: 
            https://neurodsp-tools.github.io/neurodsp/index.html

    Parameters
    ----------
    data : TYPE
        Time Course of Channel of interest. (channels x time) 
    freq_range : TYPE
        Frequency Range in which to look for bursts.
    fsample : TYPE
        Samplig Rate of time course data.
    normalise : TYPE, optional
        Function used to normalise the time course (median vs. mean). 
        The default is 'median'.
    threshold_dict : TYPE, optional
        Dict containing 'Method' for thresholding and threshold for selected method.
        Possible Methods are 'Percentile' or 'Custom'. The Latter allows to select
        exact threshold.
    min_n_cycles : TYPE, optional
        Minimum duration of bursts in cycles of lowest frequency in specified frequency range. 
        The default is 1.

    Returns
    -------
    is_burst: TYPE
        Boolean vector indicating burst evens as 1 and burst off events as 0.
    
    norm_data: TYPE
        Data after normalisation step (see above). Usefull for state metric calculation.
    '''
    # Filter Data
    filt_data = mne.filter.filter_data(data.astype(float), fsample, freq_range[0], freq_range[1])
    
    # # zscore to account for different head positions in scanner
    filt_data = zscore(filt_data,axis=0)
    
    # Get Amplitude of TC
    amp_data = np.abs(signal.hilbert(filt_data))
    
    # If 1d ad axis
    if len(amp_data.shape) == 1:
        amp_data = amp_data[np.newaxis,:]

    # Normalise by median amplitude
    if normalise == 'median':
        norm_data = amp_data - np.median(amp_data, axis = 1)
    elif normalise == 'mean':
        norm_data = amp_data - np.mean(amp_data, axis = 1)
    elif normalise == 'zscore':
        norm_data = zscore(amp_data,axis=1)
    else:
        norm_data = amp_data
        
    # # Threshold data
    # is_burst = norm_data >= np.percentile(norm_data, q=threshold, axis = 1)
    is_burst = threshold_data(norm_data, threshold_dict)
    
    # Make Sure that Bursts are at least 1 cycle - This bit is from NeuroDSP
    min_burst_samples = int(np.ceil(min_n_cycles * fsample / freq_range[0]))
    # Make here loop to loop across several channels if multi Channel Data
    is_burst = _rmv_short_periods(np.squeeze(is_burst),min_burst_samples)
    
    return is_burst.astype(bool), norm_data


def burst_time_metrics(is_burst, fsample):
    '''
    Take boolean burst On vs. Off vector and get burst starts and ends.
    Number of bursts, Burst Rate, Burst Occupancy, Lifetimes are calculated
    from these measures and returned together withs starts end ends.

    Parameters
    ----------
    is_burst : Boolean/Int
        Boolean On vs Off vector indicating presence of bursts with 1 and 
        off periods as 0.
    fsample : Int
        Sampling Rate.

    Returns
    -------
    burst_dict : dict
        Dict containing Life Times, Burst Number, Burst Occupancy, Burst Rate,
        Starts, and Ends.

    '''
    pad_bool = np.pad(is_burst.astype(int), (1), 'constant', constant_values=(0)) # pad to make it more robust for cases when starting with vector
    starts = np.where(np.diff(pad_bool) == 1)[0]-1 # -1 to account for padding
    ends= np.where(np.diff(pad_bool) == -1)[0]-1 # -1 to account for padding
    
    burst_dict = {'Life Times': (ends - starts)/fsample,
                  'Interval Times': (starts[1:]-ends[:-1])/fsample,
                  'Burst Number': len(starts),
                  'Burst Occupancy': np.sum(is_burst)/ len(is_burst),
                  'Burst Rate': len(starts)/(len(is_burst)/fsample),
                  'Starts': starts,
                  'Ends': ends}  
    return burst_dict


def custom_burst_metric(data, starts, ends, func):
    '''
    Allows to calculate custom function across each burst.
    
    Need to be adjusted for multi channel processing, ATM just working per channel.

    Parameters
    ----------
    data : TYPE
        Burst Time Course Data.
    starts : TYPE
        List with Burst starting samples.
    ends : TYPE
        List withs Burst end samples.
    func : TYPE
        Function applied to each burst.

    Returns
    -------
    List with output of func per burst.

    '''
    data = np.squeeze(data)
    custom_metric = [func(data[start_item:end_item]) for start_item, end_item in zip(starts, ends)]
    return custom_metric