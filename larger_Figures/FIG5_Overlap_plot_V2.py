#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:46:43 2022

@author: okohl

Figure 5

Visulaises Overlap Analysis.
1) Comic for Overlap Calculation is created.
2) Violine Plots comparing Overlap between Bursts and HMM states to Null Overlaps
   calculated from the overlap between bursts and randomely rolled HMM-State Time
   Courses.
3) Violines of NABBs based on states that show an decrease from the average beta
   Power when they occure are depicted in grey colors.
   
   
Importantly, in this analysis we were specifically interested in States that are
significantly MORE likely to co-occurre with bursts than randome shifted states.
Undirected t-test reveal that also a bunch of states are significantly less likely
to co-occure with states than chance.

"""

import numpy as np
import seaborn as sns
import mne
import scipy.signal as signal
from scipy.stats import zscore
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
import pickle

import sys
sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")
from plotting import p_for_annotation, split_barplot_annotate_brackets


# ----------- Object for Plotting Legends -------------

# Code for Legend borrowed from:
# https://stackoverflow.com/questions/31908982/python-matplotlib-multi-color-legend-entry

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i * 2 - handlebox.xdescent-20, 
                                          -handlebox.ydescent-4.5],
                           (width / len(orig_handle.colors)) * 2,
                           height*2, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

# --------- Functions to extract burst metrics -----------

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
                  'Burst Number': len(starts),
                  'Burst Occupancy': np.sum(is_burst)/ len(is_burst),
                  'Burst Rate': len(starts)/(len(is_burst)/fsample),
                  'Starts': starts,
                  'Ends': ends}  
    return burst_dict


# ------------------------------------------------------------------------
#%% --- Extract Motor Beta Power --- 


# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]#[1,2,3,4,5]
State_OI = 8

proj_dir = '/path/tp/proj_dir/'
outdir_plot = proj_dir + 'Results/StateMetric_GroupContrast/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
outdir_dat = proj_dir + 'Data/StateMetrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'

#%% -------------------------------------------------------
# ---- Create Data for Cartoon Figure for NABB extraction ---
irun = 7      
iSub = 50
    
print('Loading Data for Subject' + str(iSub+1))

# --- Load State and Burst Time Courses ---

# Load HMM Gammas
file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
data_in = loadmat(data_dir + file)
XX_in = data_in['subj_data'][:,17:19].T # Just Select Parcel 17 or 18

# Load Burst On Off Set Data
is_burst = np.load(data_dir + '/run' + str(irun) + '/isBurst/isBurst_Subject' + str(iSub) + '.npy')

# --- Calculate Amplitude Envelopes ---
# Get Beta Amplitude
freq_range = [18, 25]
fsample = 250

# Filter, zscore, and calculate amplitude envelope
filt_XX = mne.filter.filter_data(XX_in.astype(float), fsample, freq_range[0], freq_range[1])
XX = zscore(filt_XX,axis=1)
amp_XX = np.abs(signal.hilbert(XX,axis = 1))

# Thrshold amlitude enevelope
p75 = np.mean(np.percentile(amp_XX,75,axis=1))
 
      
# Create Three On-vs-off TCs for three States
Gamma_bin = np.zeros([8,amp_XX.shape[1]])
Gamma_bin[0,:4125] = (np.max(amp_XX,axis=0) > p75)[:4125]
Gamma_bin[1,4127:4153] = (np.max(amp_XX,axis=0) > p75)[4127:4153]
Gamma_bin[2,4155:4232] = (np.max(amp_XX,axis=0) > p75)[4155:4232]

seg_start = 4040
seg_end = 4294

# Get Burst x State Periods
starts = []
ends = []
for k in [0,1,2]:
    is_StateBurst = np.logical_and(Gamma_bin[:,seg_start:seg_end], is_burst[seg_start:seg_end])
    burst_dict = burst_time_metrics(is_StateBurst[k],fsample)
    starts.append(burst_dict['Starts'])
    ends.append(burst_dict['Ends'])



#%% -----------------------
# Load Data for Burst Overlap Plot

mdict = pickle.load(open('/path/to/proj_dir/Data/V2_Plots/Overlap_Plot_Data.p', "rb"))
df_freq = mdict['df_freq']
overlap_in = mdict['overlap_in']
p = mdict['p']
state_labels = mdict['labels']

#%% --------------------------
#  Define things for plotting

# Colors
cols = ['#BF3C37','#86C6F4','#FF9B00','#595959']
greys = ['#595959', '#262626',] #'#404040']
greys_over = ['#9e9e9e','#757575','#9e9e9e','#757575','#9e9e9e','#757575',
              '#9e9e9e','#757575','#9e9e9e','#757575','#9e9e9e','#757575',
              '#9e9e9e','#757575','#9e9e9e','#757575',]

colors =  plt.cm.tab20( np.arange(20).astype(int) )
cols = [colors[6],colors[0],colors[2]]
cols_over = [colors[7],colors[6],colors[1],colors[0],colors[3],colors[2],
             colors[15],colors[14],colors[15],colors[14],colors[15],colors[14],
             colors[15],colors[14],colors[15],colors[14]] # 17,15 for green

# Intyerval of Interest
seg_start = 4040
seg_end = 4294

# Labels
annot_labels = ['State 1', 'State 2', 'State 3', 'State 4']
statesOI = [0,1,2,3]

#%% Start Plotting

x = 1
fig = plt.figure(dpi=300, figsize=(12.8*x, 9*x), constrained_layout=False)
gs1 = fig.add_gridspec(nrows=4, ncols=2, top=.9, bottom=.6,
                       height_ratios=[1,.3,.3,.3], wspace=.2, hspace=.1)
ax = np.zeros(7, dtype=object)
ax[0] = fig.add_subplot(gs1[0, 0:3])
ax[1] = fig.add_subplot(gs1[1, 0:3])
ax[2] = fig.add_subplot(gs1[2, 0:3])
ax[3] = fig.add_subplot(gs1[3, 0:3])

gs2 = fig.add_gridspec(nrows=1, ncols=3, top=.5, bottom=.05, left=.15,
                        hspace=.1)
ax[4] = fig.add_subplot(gs2[0, 0:2])



#  --- Plot Amplitude envelope time courses ---
ax[0].plot(amp_XX[0,seg_start:seg_end], color=greys[0] ,linewidth=3)
ax[0].plot(amp_XX[1,seg_start:seg_end], color=greys[1],linewidth=3)
ax[0].axhline(p75,linestyle = '--', color='red', xmax=.96)
ax[0].set_ylim(bottom=0)

# Make Axis Pretty
ax[0].set_ylabel('Amplitude\nEnvelope', fontsize = 14, labelpad=.0)
ax[0].tick_params(left = False)
ax[0].set_yticklabels('', fontsize=10)

# Legend
legend_elements = [Line2D([0], [0], color=greys[0], lw=5, label='Left'),
                   Line2D([0], [0], color=greys[1], lw=5, label='Right')]   
ax[0].legend(handles=legend_elements,frameon=False, fontsize=14,
             bbox_to_anchor=(.995, 1), labelspacing=.1, handletextpad=0.4)

# Remove Box      
ax[0].get_xaxis().set_ticks([])
ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['right'].set_visible(False)


# Color Background
i = 0
for start, end in zip(starts,ends):
    for ind in range(len(end)):
        h = ax[0].axvspan(start[ind], end[ind],
                          color=cols[i], alpha=0.2)
    i = i + 1


# --- Add HMM State on vs off plots ---
for ind,ax_ind in enumerate([1,2,3]):
    ax[ax_ind].plot(Gamma_bin[statesOI[ind],seg_start:seg_end], color=cols[ind], linewidth=3)
    ax[ax_ind].annotate(annot_labels[ind], size=20, color=cols[ind],
                xy=(.965, -.1), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')
    ax[ax_ind].axis('off')
    ax[ax_ind].set_ylim(bottom=0)
    i = 0
    for start, end in zip(starts,ends):
        for ind in range(len(end)):
            h = ax[ax_ind].axvspan(start[ind], end[ind],
                              color=cols[i], alpha=0.2)
        i = i + 1
 


# --- Add Burst x State Overlap Figure ----
# Actual Plotting
sns.violinplot(x="State", y="Overlap", hue="Nulls",inner = "quartile",
                    data=df_freq, palette=greys_over, 
                    cut=.5 , bw=.35 ,split=True,ax = ax[4])

for ind, violin in enumerate(ax[4].findobj(PolyCollection)):
    violin.set_facecolor(cols_over[ind])
    # if ind > -1:
    #     violin.set_edgecolor(cols_over[ind])

# Set Line Colors
for l in ax[4].lines:
    l.set_linestyle('--')
    l.set_linewidth(.8)
    #l.set_color('k')
    l.set_alpha(0.8)
    
for l in ax[4].lines[1::3]:
    l.set_linestyle('-')
    l.set_linewidth(1.5)
    #l.set_color('k')
    l.set_alpha(0.8)

# Make Axis pretty
ax[4].xaxis.labelpad = 10
ax[4].set_xticklabels(state_labels, fontsize=14)
ax[4].set_xlabel('', fontsize=16)
ax[4].set_ylabel('Overlap Burst Normalised', fontsize = 18, labelpad = 8)


# Creat Patches in Legend
h = [MulticolorPatch([cols_over[1],cols_over[3],cols_over[5],cols_over[9]])] #cols_over[7],
l = ["Empirical"]

h.append(MulticolorPatch([cols_over[0],cols_over[2],cols_over[4],cols_over[8]])) # cols_over[6],
l.append("Nulls")

# create the legend
ax[4].legend(h, l, loc='upper left', prop = {'size':14}, frameon=False, 
         handler_map={MulticolorPatch: MulticolorPatchHandler()},
         bbox_to_anchor=(.024,1))

# Remove Box Around Subplot
sns.despine(ax=ax[4], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)

# Get Significance stars 

# get signifiance levels for different states!     
p_plt = []      
for iState in range(len(state_labels[:K])):
    if p[iState] < .01:
        p_plt.append(p_for_annotation(.003))
    elif p[iState] < .05:
        p_plt.append(p_for_annotation(.03))
    else:
        p_plt.append(p_for_annotation(.1))

# Parameters for significance star plottinng
heights = np.max(overlap_in,axis = 0) # Get height of violine plots
bars = np.arange(len(heights)).astype(int) # Get x Position of violines
ax_in = ax[4] # define axis to which sig stars are plotted

# Plot Starts in Violine Plot    
for ii in range(len(p_plt)):
    if ii == 2:
        barh = .17
    elif ii == 6:
        barh = .05
    else:
        barh = .01
        
    split_barplot_annotate_brackets(bars[ii], bars[ii], np.array(p_plt)[ii], bars, heights, ax_in=ax_in,
                              yerr=None, dh=.008, barh=barh, fs=20, maxasterix=None)
    
    
plt.savefig('/path/to/Fig5_Overlap.svg',transparent = True,bbox_inches="tight", 
            format='svg')