#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:48:59 2022

@author: okohl

Script Loads T-Statistics of State Metric Group Contrasts of all 30 HMM fits.
T-Statistics are plotted as beeswarm plot ber state metric allowing to assess 
whether significant differences observed in the 8 State HMM with lowest free 
energy are robust accross Model Fits and Number of States.
-> Figure 4
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:48:59 2022

@author: okohl
"""

import glmtools as glm
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
from scipy.stats import ttest_1samp
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

outdir_plot = '/home/okohl/Documents/HMM_PD_V07/Results/StateMetric_GroupContrast/'

# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [1,2,3,4,5,6,7,8,9,10]
sampling_rates = [250]
number_of_states = [8,10,12]
embeddings = [7]
sig_state_8 = 8 # State with significant effect in HMM with lowest free energy

# ---- Specify labels for loop
labels_metric = ['fractional_occupancy','mean_lifetimes', 'mean_intervaltimes','state_rates','State_GLM_BetaPower']


# Start Looping across Runs per HMMs with different number of states
tstats_acrossHMMs = np.zeros([len(runs),len(labels_metric),len(number_of_states)])
fe_rank_acrossHMMs = np.zeros([len(runs),len(number_of_states)])
for fsample in sampling_rates:
    for emb in embeddings:
        for K_ind,K in enumerate(number_of_states):
            
            # # For 10K sig state because state matching is not working very well!!!!!
            # if K == 8:
            #     sig_state = 1
            # elif K == 10:
            #     sig_state = 6
            # elif K == 12:
            #     sig_state = 10
                
            if K == 8:
                sig_state = sig_state_8
            elif K > 8:
                # Load state assignments from State matching
                StateMatching_file = '/path/to/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(K) + '_to_8K_RefHMM_matching.mat'
                sig_state =  loadmat(StateMatching_file)['assig'][:,sig_state_8-1]
            
            # Load state assignments from State matching // Just for State significantly matching to State significant in 8K HMM 
            StateMatching_file = '/path/to/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(K) + '/StateMatching.mat'
            state_OI = loadmat(StateMatching_file)['assig_all'][:,sig_state-1]

            
            # --- Load TStatistics from Group Comp ---
            tstats_all = np.zeros([len(runs),len(labels_metric)])
            for irun in runs:
                            
                            print('Running Analysis for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
                                  
                            indir = '/path/to/Data/StateMetrics/ds' + \
                                str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                            
                            indir_dat = '/path/to/Data/StateMetrics/ds' + \
                                str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                                
                            #% Start Loop Calculating GLM for each State Metric
                            for iMetric, metric in enumerate(labels_metric):
                            
                                tstats_in = pickle.load(open(indir_dat + labels_metric[iMetric] +
                                                    "_GroupComp_GLM.mat", "rb"))['tstats']
                                
                                tstats_all[irun-1,iMetric] = tstats_in[0,state_OI[irun-1]-1] # -1 Python indexing
                                
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


# Set Axis Labels and Ticks
ax.set_xticklabels(['Fractional\nOccupancy','Life\nTimes','Interval\nTimes','State\nRate','Mean\nBeta Power'])
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
 
# Set y=0 Line    
ax.axhline(0, color = 'grey', linestyle = '--', linewidth = .5)
    
# Remove Box Around Subplot
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

# Save Data
plt.savefig(outdir_plot + 'across_K_runs_State' + str(sig_state_8) + '_Group_Contrast_Robustness_Kcolor.svg',
            bbox_inches='tight', transparent = False, format = 'svg')
            



# # ------ Results for further Plotting ------
# mdict = {'Data': df_tstats}
# pickle.dump(mdict, open( "/home/okohl/Documents/HMM_PD_V07/Data/V2_Plots/State" + str(sig_state_8) +
#                         "_GroupContrast_Robustness.mat", "wb"))
