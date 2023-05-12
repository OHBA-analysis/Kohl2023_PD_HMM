#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:46:43 2022

@author: okohl

Make plot depicting Sensorimotor State Beta Power Association and State Metric
Group Differences

-> Figure 4
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import (MultipleLocator)
from matplotlib import ticker
import pickle

import sys

sys.path.append("/path/to/helpers/")

from plotting import get_colors



# ------------------- Functoions -------------------

def get_group_vect():
    # Get Vector indicateing which participants are patients
    # TTo do: his can be done a lot easier!
    
    subIDs = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19',
                     '51','52','53','54','56','57','58','59','61','62','63','64','65','66','67','68','69','70','71',
                     '101','102','103','104','105','106','107','109','110','116','117','118',
                     '151','153','154','155','156','157','159','160','161','162','163','164','165','166','167','168','169','170']

    HCsub = np.array(range(0, subIDs.index('71')+1))
    PDsub = np.array(range(HCsub[-1]+1, subIDs.index('170')+1))
    allSub = np.hstack([HCsub, PDsub])
    
    # 1s for HCs and 2s for PD patients
    group_vect = np.ones_like(allSub)
    group_vect[allSub > len(HCsub)-1] = 2
    
    return group_vect



#%% -------------------------------------------------
# Set Parameters

# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fsample = 250
irun = 7
State_OI = 8

    
# ------------------------------
# colors
col_dict = get_colors()
PDvsHC_palette = col_dict['PDvsHC_palette']
proj_colors = plt.cm.Reds(np.linspace(.3, 1, 5))


# -----------------------------
# Load Data for Porjection and Correl Plots
mdict_corr = pickle.load(open('/path/to/FO_Power_correl.p', "rb"))
mdict_proj = pickle.load(open('/path/to/FO_Power_Projections.p', "rb"))
mdict_robust = pickle.load(open( "/path/to/State8_GroupContrast_Robustness.mat", "rb"))

#------------------------------
# Set Dirs for Data Loading of State Metric Plots
HMM_indir = '/path/to//Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

GLM_indir = '//path/to//Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                

# --------------------------------------------
# Define Labels for State Metric Plots
# To Load State Metrics
labels_metric = ['fractional_occupancy','mean_lifetimes', 'mean_intervaltimes','state_rates', 'State_GLM_BetaPower']

# For Plotting
labels = ['State \n 1', 'State \n 2', 'State \n 3', 'State \n 4', 'State \n 5', 'State \n 6',
          'State \n 7', 'State \n 8', 'State \n 9', 'State \n 10', 'State \n 11', 'State \n 12']
labels_plot = ['Fractional Occupancy', 'Lifetimes (sec)',
               'Interval Times (sec)', 'State Rates', 'Motor Beta Power (a.u.)']

# -------------------------------------
# Load other vars for State Metric Plots

# Load State Sorting vector
sort_states = np.load('/path/to//Data/sortingVector/ClusterPowSorting.npy')

# Get Group Vector
group_vect = get_group_vect()


#%% ------------------------------------------------------
# Start Plotting

# --- Set up Grid ---

x = 1
fig = plt.figure(dpi=300, figsize=(15*x, 12*x), constrained_layout=False)
gs1 = fig.add_gridspec(nrows=1, ncols=2, top=.98, bottom=.65, wspace=.2, hspace=.1)

ax = np.zeros(8, dtype=object)
ax[0] = fig.add_subplot(gs1[0, 0])
ax[1] = fig.add_subplot(gs1[0, 1])

gs2 = fig.add_gridspec(nrows=1, ncols=5, top=.52, bottom=.37, wspace=1,
                        hspace=.1)
ax[2] = fig.add_subplot(gs2[0, 0])
ax[3] = fig.add_subplot(gs2[0, 1])
ax[4] = fig.add_subplot(gs2[0, 2])
ax[5] = fig.add_subplot(gs2[0, 3])
ax[6] = fig.add_subplot(gs2[0, 4])

gs3 = fig.add_gridspec(nrows=1, ncols=5, top=.3, bottom=.02, wspace=1,
                        hspace=.1)
ax[7] = fig.add_subplot(gs3[0, 1:4])


#%% --- Fo x Beta Power Corrleation ---

dat_plot = mdict_corr['dat_plot']
PDs = mdict_corr['PDs']
r = mdict_corr['r']
p = mdict_corr['p']


# --- Start Plotting ---

sns.set_style("ticks")
sns.regplot(x="Beta Power", y="Fractional Occupancy", data=dat_plot, ax = ax[0], color = PDvsHC_palette[1] )
ax[0].scatter(dat_plot['Beta Power'][PDs],dat_plot['Fractional Occupancy'][PDs], color = PDvsHC_palette[0])

# Set axis Labls and Ticks
ax[0].set_ylabel(labels[-1], fontsize = 12, labelpad = 10)
ax[0].set_xlabel('Motor Beta Power (a.u.)', fontsize = 16, labelpad = 10)
ax[0].set_ylabel('Fractional Occupancy', fontsize = 16, labelpad = 7)

ax[0].tick_params(axis='x', labelsize= 12)
ax[0].tick_params(axis='y', labelsize= 12)

# Format y-Axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax[0].xaxis.set_major_formatter(formatter)

# Add Text to plot
if p < .001:
    ax[0].text(.02,.98,'r = ' + str(np.round(r,2)) + ', p < 0.001',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax[0].transAxes,
            fontsize = 14)
else:
    ax[0].text(.02,.98,'r = ' + str(np.round(r,2)) + ', p = ' + str(np.round(p,3)),
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax[0].transAxes,
            fontsize = 14)

# Remove Box Around Subplot
sns.despine(ax=ax[0], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)


#%% Projection plot

# Load Data for Projection plot
model_all = mdict_proj['model_all']
freqs = mdict_proj['freqs']

# Specify Parameters
nsteps = 5
iState = 0
contrast = 5
ymax = 0.0008
rois = [9,17,18]

# Get Prediction
proj, step_labels = model_all[iState].project_range(contrast, nsteps=nsteps)


# --- Start Plotting ---
for iSteps in range(nsteps):
    ax[1].plot(freqs,np.mean(proj[iSteps,:,rois].T,1), color = proj_colors[iSteps], linewidth = 2)

# Legend
leg = ax[1].legend(np.round(step_labels,2),fontsize = 12, fancybox = True, title='FO', 
             frameon = False, loc='upper right', bbox_to_anchor = (1.15,1))
leg.get_title().set_fontsize('14')

# Axis Limits
ax[1].set_xlim(left = 1)    
ax[1].set_ylim(bottom = 0, top = ymax) 

# Axis Labels
ax[1].set_ylabel('Power (a.u.)', fontsize = 16, labelpad = 7)
ax[1].set_xlabel('Frequency (Hz)', fontsize = 16, labelpad = 10) 

# Set X axis Ticks
ax[1].xaxis.set_major_locator(MultipleLocator(5))
ax[1].xaxis.set_major_formatter('{x:.0f}')

ax[1].yaxis.set_major_locator(MultipleLocator(.0002))  
ax[1].yaxis.set_minor_locator(MultipleLocator(.0001))

ax[1].tick_params(axis='both', which='major', labelsize=12)

# Format y-Axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax[1].yaxis.set_major_formatter(formatter)

# Add lines indicating freqs to plot
ax[1].axvline(5, color = 'grey', linestyle = '--', linewidth = .5)
ax[1].axvline(13, color = 'grey', linestyle = '--', linewidth = .5)
ax[1].axvline(30, color = 'grey', linestyle = '--', linewidth = .5)

# Remove Top and Right box line
sns.despine(ax = ax[1], left=False, bottom=False)



#%% State Metric Group Contrast Plots

tck_dist = [.03,.01,.1, .4, .002] # To make sure that we have 5 y-ticks per plot

for iMetric, metric in enumerate(labels_metric):
    
    # Load importantb vars from GLM
    GLM_in = pickle.load(open( GLM_indir + metric + "_GroupComp_GLM.mat", "rb" ) ) # adjust
    tstats = GLM_in['tstats'][:,sort_states]
    thresh = GLM_in['thresh']
    
    # Load State Metrics inferred with HMM-MAR Toolbox
    in_data = loadmat(HMM_indir + metric + ".mat")['out'][:,sort_states]
    df_states = pd.DataFrame(in_data, columns = labels[:K]) # -1 because of Python indexing
    
    # DF for plotting
    df_fact = pd.DataFrame(np.vstack([
                                np.ones([1, len(df_states)]),
                                group_vect,
                                np.squeeze(in_data[:,0])]).T,
                            columns=['State', 'Group',metric])
     
    # ---  Violine Plot ---    
    sns.violinplot(x="State", y=metric, hue="Group",inner = "quartile",
                        data=df_fact, palette=PDvsHC_palette, 
                        cut=.5 , bw=.35 ,split=True,ax = ax[2 + iMetric])
    
    for ind, violin in enumerate(ax[2 + iMetric].findobj(PolyCollection)):
        violin.set_facecolor(PDvsHC_palette[ind])
        if ind > 7:
            violin.set_edgecolor(PDvsHC_palette[ind])   
    
    # Set Line Colors
    for l in ax[2 + iMetric].lines:
        l.set_linestyle('--')
        l.set_linewidth(.8)
        l.set_color('#AEAEAE')
        l.set_alpha(0.8)
        
    for l in ax[2 + iMetric].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(2)
        l.set_color('#AEAEAE')
        l.set_alpha(0.8)
    
    # Labels and Ticks    
    ax[2 + iMetric].set_xlabel('')
        
    ax[2 + iMetric].set_xticklabels('', fontsize=16) #becasue of np.arrang does not include end of specified range
    ax[2 + iMetric].set_ylabel(labels_plot[iMetric], labelpad = 7)
    ax[2 + iMetric].yaxis.label.set_size(16)
    ax[2 + iMetric].tick_params(axis='both', which='major', bottom=False, top=False)
    ax[2 + iMetric].tick_params(axis='both', which='minor', bottom=False, top=False)
    
    ax[2 + iMetric].yaxis.set_major_locator(MultipleLocator(tck_dist[iMetric]))
    
    # Format y-Axis
    if iMetric == 4:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1)) 
        ax[2+iMetric].yaxis.set_major_formatter(formatter)
    
    # Lagend
    if iMetric == 4:
        ax[2 + iMetric].legend(prop = {'size':16}, frameon=False, loc='upper right' , bbox_to_anchor = (1.7,1))
        leg = ax[2 + iMetric].axes.get_legend()
        new_title = ''
        leg.set_title(new_title)
        new_labels = ['HC', 'PD']
        for t, l in zip(leg.texts, new_labels): t.set_text(l)
    else:
        ax[2 + iMetric].legend([],frameon=False)
    
    # Remove Box Around Subplot
    sns.despine(ax=ax[2 + iMetric], top=True, right=True, left=False,
                bottom=False, offset=None, trim=True)
    
    
    # ---- Get Significance stars in violine plots -----
    
    # get asterix according to significance level    
    if abs(tstats[:,iState]) > thresh[1]:
        pval = '**'
    elif abs(tstats[:,iState]) > thresh[0]:
        pval = '*'
    else:
        pval = ''
    
    # Add asterix
    x_position = -.03
    
    if iMetric == 4:
        y_position = max(in_data[:,0]) * 1.063
    else:
        y_position = max(in_data[:,0]) * 1.045
        
    ax[2 + iMetric].text(x=x_position, y=y_position, s=pval, zorder=10, size=16 )
    


#%% Plotting Tstatistics of Metric Group Contrast colored by K

# Load Data
df_tstats = mdict_robust['Data']
cols = plt.cm.Reds(np.linspace(.4, 1, 3))
          
# --- Plot ---
points = sns.swarmplot(data=df_tstats, x="Metric", y="T-Statistic", hue='K',
              palette=cols, size=5, ax=ax[7])

sns.boxplot(data=df_tstats, x="Metric", y="T-Statistic", width = .5,
            color='white', ax=ax[7])

# Set Axis Labels and Ticks
ax[7].set_xticklabels(['Fractional\nOccupancy','Life\nTimes','Interval\nTimes','State\nRate','Mean\nBeta Power'])
ax[7].set_xlabel('')
ax[7].set_ylabel('T-Statistic', fontsize=16)
ax[7].tick_params(axis='x', which='major', labelsize=16)
ax[7].tick_params(axis='y', which='major', labelsize=12) 

# Legend
ax[7].legend(bbox_to_anchor=(1.1,1),frameon=False, prop={'size': 12})
leg = ax[7].axes.get_legend()
new_title = 'No. of States'
leg.set_title(new_title)
leg.get_title().set_fontsize('14')

new_labels = ['8','10','12']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)
    
leg.legendHandles[0]._sizes = [100]
leg.legendHandles[1]._sizes = [100]
leg.legendHandles[2]._sizes = [100]
  
# Make Line at y = 0  
ax[7].axhline(0, color = 'grey', linestyle = '--', linewidth = .5)
    
# Remove Box Around Subplot
sns.despine(ax=ax[7], top=True, right=True, left=False, bottom=False)

# Save Figure
plt.savefig('/path/to/Results/V2_Plots/Fig3_StateMetrics_large.svg',transparent = True,bbox_inches="tight", 
            format='svg')