#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:23:05 2023

@author: okohl

Figure 6

NABB Metrics Group Contrast visualised as violine plots.
Results of significant group differences of Sensorimotor NABB and frontal NABB
are plotted.

Signifcance calculate with max-Tstatistic Permutation tests is indicated by asterixes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator)
import pickle
import pandas as pd
import seaborn as sns

import sys

sys.path.append("/path/to/helpers/")

from plotting import get_colors
    

# ---------------------------------------------------
# %% Make Plots for NABB Group Comparison GLMs
# ---------------------------------------------------
# Script creates split Violin Plots to depict group 
# differences in NABB Metrics.
#
# Significant Differences are marked with ** or *
# ----------------------------------------------------



# colors
cols_dict = get_colors()
PDvsHC_palette = cols_dict['PDvsHC_palette']

# --- Get Subjects that are in Source space analysis ---
subIDs = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19',
                 '51','52','53','54','56','57','58','59','61','62','63','64','65','66','67','68','69','70','71',
                 '101','102','103','104','105','106','107','109','110','116','117','118',
                 '151','153','154','155','156','157','159','160','161','162','163','164','165','166','167','168','169','170']

HCsub = np.array(range(0, subIDs.index('71')+1))
PDsub = np.array(range(HCsub[-1]+1, subIDs.index('170')+1))
allSub = np.hstack([HCsub, PDsub])

group_vect = np.ones_like(allSub)
group_vect[allSub > len(HCsub)-1] = 2


# --- Define Labels ---
# To Load State Metrics
labels_metric = ['Fractional Occupancy', 'Mean Lifetimes', 'Mean Interval Times','Burst Rates', 'Mean Amplitude']


# For Plotting
labels = ['State \n 1', 'State \n 2', 'State \n 3', 'State \n 4', 'State \n 5', 'State \n 6',
          'State \n 7', 'State \n 8', 'State \n 9', 'State \n 10', 'State \n 11', 'State \n 12']
labels_plot = ['Fractional Occupancy', 'Lifetimes (sec)',
               'Interval Times (sec)', 'State Rates', 'Motor Beta Power (a.u.)']
 
# Load State Sorting vector
sort_states = np.load('/home/okohl/Documents/HMM_PD_V07/Data/sortingVector/ClusterPowSorting.npy')


# ----------------------------------
# --- Load Data for Correl Plots ---
mdict_FO = pickle.load(open( "/path/to/V2_Plots/fractional_occupancyNAAB_correl_Data.mat", "rb"))
mdict_LT = pickle.load(open( "/path/to/V2_Plots/V2_Plots/mean_lifetimesNAAB_correl_Data.mat", "rb"))


# ------------------------------
# --- Start Loop across HMMs ---

# Set Params
fs = 250
number_of_states = [8]
nrepeate = [7]#[1,2,3,4,5]

tck_dist = [.015,.005,.2, .3, .001]

# ------------------------
# Set up plot
x = 1
fig = plt.figure(dpi=300, figsize=(14*x, 7*x), constrained_layout=False)
gs1 = fig.add_gridspec(nrows=2, ncols=6, top = .95, bottom = .05, 
                       left=.05, right=.45, wspace=2.2, hspace=.45)

ax = np.zeros(9, dtype=object)
ax[0] = fig.add_subplot(gs1[0, 0:2])
ax[1] = fig.add_subplot(gs1[0, 2:4])
ax[2] = fig.add_subplot(gs1[0, 4:6])
ax[3] = fig.add_subplot(gs1[1, 1:3])
ax[4] = fig.add_subplot(gs1[1, 3:5])

gs2 = fig.add_gridspec(nrows=2, ncols=3, top = .95, bottom = .05,
                       left=.58, right=.95, wspace=1.2, hspace=.45, width_ratios=[1.7,1,1])

ax[5] = fig.add_subplot(gs2[0, 0])
ax[6] = fig.add_subplot(gs2[1, 0])
ax[7] = fig.add_subplot(gs2[0, 1:3])
ax[8] = fig.add_subplot(gs2[1, 1:3])

for K in number_of_states: # Loop over different states
    for irun in nrepeate:  # Loop over different HMM inferences
       
        # --- Set Dirs ---
        # Set dirs
        proj_dir = '/path/tp/proj_dir/'
        indir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        outdir_plot = proj_dir + 'Results/State_x_Burst_Metric_GroupContrast/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        data_outdir = proj_dir + 'Data/Burst_x_State_Metrics_GroupContrast/ds250/K' + str(K) + '/run' + str(irun) + '/'

        # --- Load State Metrics ---
        allMetrics = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metrics']
                
        for iMetric, metric in enumerate(labels_metric): # -1 to exclude switch rates
            
            # Store vars from GLM
            GLM_in = pickle.load(open(data_outdir + metric + 
                                      "_StateBurst_GroupComp_GLM.mat", "rb"))
            tstats = GLM_in['tstats'][sort_states]
            thresh = GLM_in['thresh']
            p = [GLM_in['p'][i] for i in sort_states]
            
            metric_in = np.squeeze(allMetrics[iMetric][:,sort_states])
            
            # DF for plotting
            df_fact = pd.DataFrame(np.vstack([
                                        np.ones([1, len(metric_in)]),
                                        group_vect,
                                        np.squeeze(metric_in[:,0])]).T,
                                    columns=['State', 'Group',metric])
         
             
            # ---  Violine Plot ---
            sns.violinplot(x="State", y=metric, hue="Group",inner = "quartile",
                                data=df_fact, palette=PDvsHC_palette, 
                                cut=.5 , bw=.35 ,split=True,ax = ax[iMetric])
            
            for ind, violin in enumerate(ax[iMetric].findobj(PolyCollection)):
                violin.set_facecolor(PDvsHC_palette[ind])
                if ind > 7:
                    violin.set_edgecolor(PDvsHC_palette[ind])
            
            # Set Line Colors
            for l in ax[iMetric].lines:
                l.set_linestyle('--')
                l.set_linewidth(.8)
                l.set_color('#AEAEAE')
                l.set_alpha(0.8)
                
            for l in ax[iMetric].lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(2)
                l.set_color('#AEAEAE')
                l.set_alpha(0.8)
            
            
            # Labels and Ticks    
            ax[iMetric].set_xlabel('')
                
            ax[iMetric].set_xticklabels('', fontsize=16) #becasue of np.arrang does not include end of specified range
            ax[iMetric].set_ylabel(labels_plot[iMetric])
            ax[iMetric].yaxis.label.set_size(16)
            ax[iMetric].tick_params(axis='both', which='major', bottom=False, top=False)
            ax[iMetric].tick_params(axis='both', which='minor', bottom=False, top=False)
            
            ax[iMetric].yaxis.set_major_locator(MultipleLocator(tck_dist[iMetric]))
            
            # Format y-Axis
            if iMetric == 4:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True) 
                formatter.set_powerlimits((-1,1)) 
                ax[iMetric].yaxis.set_major_formatter(formatter)
            
            # Legend
            if iMetric == 4:
                ax[iMetric].legend(prop = {'size':16}, frameon=False, loc='upper right' , bbox_to_anchor = (1.7,1))
                leg = ax[iMetric].axes.get_legend()
                new_title = ''
                leg.set_title(new_title)
                new_labels = ['HC', 'PD']
                for t, l in zip(leg.texts, new_labels): t.set_text(l)
            else:
                ax[iMetric].legend([],frameon=False)
            
            # Remove Box Around Subplot
            sns.despine(ax=ax[iMetric], top=True, right=True, left=False,
                        bottom=False, offset=None, trim=True)

            
            # ---- Get Significance stars in violine plots -----
            
            # get asterix according to significance level    
            if abs(tstats[0]) > thresh[1]:
                pval = '**'
            elif abs(tstats[0]) > thresh[0]:
                pval = '*'
            else:
                pval = ''
            
            # Add asterix
            x_position = -.07
            
            if iMetric == 4:
                y_position = max(metric_in[:,0]) * 1.063
            else:
                y_position = max(metric_in[:,0]) * 1.045
                
            ax[iMetric].text(x=x_position, y=y_position, s=pval, zorder=10, size=16 )
            

# State 3 NABB Plot
            
labels_metric = ['Fractional Occupancy', 'Mean Lifetimes']
tck_dist = [.015,.01]   
            
for K in number_of_states: # Loop over different states
    for irun in nrepeate:  # Loop over different HMM inferences
       
        # --- Set Dirs ---
        # Set dirs
        proj_dir = '/path/to/proj_dir/'
        indir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        outdir_plot = proj_dir + 'Results/State_x_Burst_Metric_GroupContrast/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        data_outdir = proj_dir + 'Data/Burst_x_State_Metrics_GroupContrast/ds250/K' + str(K) + '/run' + str(irun) + '/'

        # --- Load State Metrics ---
        allMetrics = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metrics']
                
        for iMetric, metric in enumerate(labels_metric): # -1 to exclude switch rates
            
            # Store vars from GLM
            GLM_in = pickle.load(open(data_outdir + metric + 
                                      "_StateBurst_GroupComp_GLM.mat", "rb"))
            tstats = GLM_in['tstats'][sort_states]
            thresh = GLM_in['thresh']
            p = [GLM_in['p'][i] for i in sort_states]
            
            metric_in = np.squeeze(allMetrics[iMetric][:,sort_states] )# -1 because of Python indexing
            
            # DF for plotting
            df_fact = pd.DataFrame(np.vstack([
                                        np.ones([1, len(metric_in)]),
                                        group_vect,
                                        np.squeeze(metric_in[:,2])]).T,
                                    columns=['State', 'Group',metric])
         
             
            # ---  Violine Plot ---
            sns.violinplot(x="State", y=metric, hue="Group",inner = "quartile",
                                data=df_fact, palette=PDvsHC_palette, 
                                cut=.5 , bw=.35 ,split=True,ax = ax[iMetric + 5])
            
            for ind, violin in enumerate(ax[iMetric + 5].findobj(PolyCollection)):
                violin.set_facecolor(PDvsHC_palette[ind])
                if ind > 7:
                    violin.set_edgecolor(PDvsHC_palette[ind])
            
            # Set Line Colors
            for l in ax[iMetric + 5].lines:
                l.set_linestyle('--')
                l.set_linewidth(.8)
                l.set_color('#AEAEAE')
                l.set_alpha(0.8)
                
            for l in ax[iMetric + 5].lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(2)
                l.set_color('#AEAEAE')
                l.set_alpha(0.8)
            
            
            # Labels and Ticks    
            ax[iMetric + 5].set_xlabel('')
                
            ax[iMetric + 5].set_xticklabels('', fontsize=16) #becasue of np.arrang does not include end of specified range
            ax[iMetric + 5].set_ylabel(labels_plot[iMetric])
            ax[iMetric + 5].yaxis.label.set_size(16)
            ax[iMetric + 5].tick_params(axis='both', which='major', bottom=False, top=False)
            ax[iMetric + 5].tick_params(axis='both', which='minor', bottom=False, top=False)
            
            ax[iMetric + 5].yaxis.set_major_locator(MultipleLocator(tck_dist[iMetric]))
            
            # Format y-Axis
            if iMetric == 4:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True) 
                formatter.set_powerlimits((-1,1)) 
                ax[iMetric].yaxis.set_major_formatter(formatter)
            

            ax[iMetric + 5].legend([],frameon=False)
            
            # Remove Box Around Subplot
            sns.despine(ax=ax[iMetric + 5], top=True, right=True, left=False,
                        bottom=False, offset=None, trim=True)

            
            # ---- Get Significance stars in violine plots -----
            
            # get asterix according to significance level    
            if abs(tstats[2]) > thresh[1]:
                pval = '**'
            elif abs(tstats[2]) > thresh[0]:
                pval = '*'
            else:
                pval = ''
            
            # Add asterix
            x_position = -.06
            
            if iMetric == 0:
                y_position = max(metric_in[:,2]) * 1.03
            else:
                y_position = max(metric_in[:,2]) * 1.025
                
            ax[iMetric + 5].text(x=x_position, y=y_position, s=pval, zorder=10, size=16 )
            
            
            
#%% State 3 NABB UPDRS correlation


# ------- Fractional Occupancy ------------

# --- Set Parameters Specifying for which HMM tests are calculated -------
dat_plot = mdict_FO['Data']
tstat = mdict_FO['tstat']
p = mdict_FO['p']

# Make Scatter Plot with regression line                
sns.set_style("ticks")
sns.regplot(x="Bradykinesia/Rigidity", y="Interval Times", data=dat_plot, ax = ax[7], color = PDvsHC_palette[1])

# Axis Labels and Ticks
ax[7].set_ylabel('Fractional Occupancy', fontsize = 16, labelpad = 10)
ax[7].set_xlabel('Bradykinesia/\nRigidity', fontsize = 16, labelpad = 5)

ax[7].tick_params(axis='x', labelsize= 12)
ax[7].tick_params(axis='y', labelsize= 12)

ax[7].yaxis.set_major_locator(MultipleLocator(.01))

# Text showing statistical values
ax[7].text(.95,.98,'T-Stat = ' + str(np.round(tstat,2)[0]) + ', p = ' + str(np.round(p,2)),
        horizontalalignment='right',
        verticalalignment='top',
        transform = ax[7].transAxes,
        fontsize = 14)

# Remove Box Around Subplot
sns.despine(ax=ax[7], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)



# ------- Life Times ------------

# --- Set Parameters Specifying for which HMM tests are calculated -------
dat_plot = mdict_LT['Data']
tstat = mdict_LT['tstat']
p = mdict_LT['p']

 # Make Scatter Plot with regression line                
sns.set_style("ticks")
sns.regplot(x="Bradykinesia/Rigidity", y="Interval Times", data=dat_plot, ax = ax[8], color = PDvsHC_palette[1])

# Axis Labels and Ticks
ax[8].set_ylabel('Lifetimes (sec)', fontsize = 16, labelpad = 10)
ax[8].set_xlabel('Bradykinesia/\nRigidity', fontsize = 16, labelpad = 5)

ax[8].tick_params(axis='x', labelsize= 12)
ax[8].tick_params(axis='y', labelsize= 12)

ax[8].yaxis.set_major_locator(MultipleLocator(.005))

# Text showing statistical values
ax[8].text(.95,.98,'T-Stat = ' + str(np.round(tstat,2)[0]) + ', p = ' + str(np.round(p,2)),
        horizontalalignment='right',
        verticalalignment='top',
        transform = ax[8].transAxes,
        fontsize = 14)

# Remove Box Around Subplot
sns.despine(ax=ax[8], top=True, right=True, left=False,
        bottom=False, offset=None, trim=False)

            
#Save Plot    
plt.savefig('/path/to/V2_Plots/Fig6_NABB_Metric_GroupContrast_V2.svg',
              transparent=True, bbox_inches="tight",format='svg')
            
