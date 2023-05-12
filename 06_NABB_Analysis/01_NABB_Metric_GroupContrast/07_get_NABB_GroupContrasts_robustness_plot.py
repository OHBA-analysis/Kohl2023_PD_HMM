#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:48:59 2022

@author: okohl

Plot t-stats of PD vs HC group contrasts of all state metrics of all HMM runs.
For each State Metric the T-statistics of the group contrasts of all HMMs are
plotted.

T-Statistics from HMMs inferring different numbers of states are color coded
differently.

-> SI7

"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_1samp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

outdir_plot = '/path/to//Results/State_x_Burst_Metric_GroupContrast/'

# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [1,2,3,4,5,6,7,8,9,10]
sampling_rates = [250]
number_of_states = [8,10,12]
embeddings = [7]
sig_state_8 = 8 # State with significant effect in HMM with lowest free energy

# ---- Specify labels for loop
labels_metric = ['Fractional Occupancy','Mean Lifetimes', 'Mean Interval Times','Burst Rates','Mean Power']


# Start Looping across Runs per HMMs with different number of states
tstats_acrossHMMs = np.zeros([len(runs),len(labels_metric),len(number_of_states)])
for fsample in sampling_rates:
    for emb in embeddings:
        for K_ind,K in enumerate(number_of_states):
            
            # Load state assignments from State matching
            if K == 8:
                sig_state = sig_state_8
            elif K > 8:
                StateMatching_file = '/home/okohl/Documents/HMM_PD_V07/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(K) + '_to_8K_RefHMM_matching.mat'
                sig_state =  loadmat(StateMatching_file)['assig'][:,sig_state_8-1]
            
            # Load state assignments from State matching // Just for State significantly matching to State significant in 8K HMM 
            StateMatching_file = '/home/okohl/Documents/HMM_PD_V07/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(K) + '/StateMatching.mat'
            state_OI = loadmat(StateMatching_file)['assig_all'][:,sig_state-1]
            
            # --- Load TStatistics from Group Comp ---
            tstats_all = np.zeros([len(runs),len(labels_metric)])
            for irun in runs:
                            
                            print('Running Analysis for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
                                  
                            indir = '/home/okohl/Documents/HMM_PD_V07/Data/Burst_x_State_Metrics_GroupContrast/ds' + \
                                str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                            
                            indir_dat = '/home/okohl/Documents/HMM_PD_V07/Data/Burst_x_State_Metrics_GroupContrast/ds' + \
                                str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                                
                            #% Start Loop Calculating GLM for each State Metric
                            for iMetric, metric in enumerate(labels_metric):
                            
                                tstats_in = pickle.load(open(indir_dat + labels_metric[iMetric] +
                                                    "_StateBurst_GroupComp_GLM.mat", "rb"))['tstats']
                                
                                tstats_all[irun-1,iMetric] = tstats_in[state_OI[irun-1]-1] # -1 Python indexing
                                
            tstats_acrossHMMs[:,:,K_ind] = tstats_all
                    
# --- Bring Tstats into shape for plotting ---                   
df_tstats = []
for k in range(len(number_of_states)):
    for i in range(len(labels_metric)):
        metric_tmp = np.ones([1, len(runs)])*(i+1)
        K_tmp = np.ones([1, len(runs)])*(k+1)
        t_tmp = tstats_acrossHMMs[:,i,k]
        dat_tmp = np.vstack([np.squeeze(metric_tmp),np.squeeze(t_tmp), np.squeeze(K_tmp)]).T
        df_tstats.append(dat_tmp)  

df_tstats = np.vstack(df_tstats)
df_tstats = pd.DataFrame(df_tstats, columns=['Metric','T-Statistic','K'])


# --- 1 Sample Ttest to check whether t values significantly different from 0 ---
statistic = np.zeros(len(labels_metric))
pvalue = np.zeros(len(labels_metric))

for iMetric in range(len(labels_metric)):
    sample_observation = df_tstats['T-Statistic'][df_tstats['Metric'] == iMetric+1]
    [statistic[iMetric],pvalue[iMetric]] = ttest_1samp(sample_observation,0)


# --- Plotting Tstatistics of Metric Group Contrast colored by K ---

# Get Colors
cols = plt.cm.Reds(np.linspace(.4, 1, len(number_of_states)))
          
# plot
sns.set_style("ticks")
fig = plt.figure(dpi=300)
ax = fig.add_subplot()
points = sns.swarmplot(data=df_tstats, x="Metric", y="T-Statistic", hue='K',
              palette=cols, size=5, ax=ax)

sns.boxplot(data=df_tstats, x="Metric", y="T-Statistic", width = .5,
            color='white', ax=ax)


ax.set_xticklabels(['Fractional\nOccupancy','Life\nTimes','Interval\nTimes','State\nRate','Beta\nPower'])
ax.set_xlabel('')
ax.set_ylabel('T-Statistic', fontsize=16)
ax.tick_params(axis='x', which='major', labelsize=14)
ax.tick_params(axis='y', which='major', labelsize=12) 

# Legend
ax.legend(bbox_to_anchor=(1.1,1),frameon=False, prop={'size': 6})
leg = ax.axes.get_legend()
new_title = 'K'
leg.set_title(new_title)
new_labels = ['8','10','12']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)
    
ax.axhline(0, color = 'grey', linestyle = '--', linewidth = .5)
    
# Remove Box Around Subplot
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

plt.savefig(outdir_plot + 'across_K_runs_Burst_x_State' + str(sig_state_8) + '_Group_Contrast_Robustness_Kcolor.svg',
            bbox_inches='tight', transparent = False, format = 'svg')
            
