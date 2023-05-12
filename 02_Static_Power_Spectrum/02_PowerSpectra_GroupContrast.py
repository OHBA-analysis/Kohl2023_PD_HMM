#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:58:37 2022

@author: okohl


1) Contrast Power between HCs and PD patients with Cluster-based Permutation tests.
2) Plot Power Spectra and highlight significant clusters. -> Figure 2

"""


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import mne
import seaborn as sns

import sys
sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")
from plotting import get_colors


# --- Set Parameters and Dirs ----
proj_dir = 'path/to/proj_dir/'
Pow_dir = proj_dir + 'Data/staticSpectra/'
GLM_dir = proj_dir + 'Data/static_Group_GLM/'
plot_dir = proj_dir + 'Results/StaticPower/2nd_Level/'

nSub = 67
fs = 250
HC_end = 36


# --- Indetify Participants to include ---
# Participants, for which confounds (Age, Gender, Education, and Handedness) are
# not collected, are not included to make sure that same participants are used for all
# Analyses.
behav_in = 'path/to/behavioral_data/'
df = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')
df_in = df[['Group','Handedness','Gender','Age','Education']]
in_ind = np.prod(df_in.notna().values,axis = 1)
in_ind = np.ma.make_mask(in_ind)


# --- load Subjects' Power Spectra ---
in_data = pickle.load(open( Pow_dir + "zscore_pow_all.dat", "rb" ) )
freqs = in_data['Freqs']

# remove freqs < 2Hz
keeps = np.where(freqs>=2)[0]
freqs = freqs[keeps]

# Number of subs with missing data
outSub = sum(~in_ind)

# Grab Power Spectra included in analysis
copes = in_data['Pow'][in_ind,:,:][:,keeps,:]


#%% ---- Calculate Cluster Permutation Tests and bring into long format  ---

# --- Average across motor parcels ---
mean_pow = np.mean(copes[:,:,17:19], axis=2)

# --- Cluster-based Permutation test ---
perm_in = [mean_pow[:HC_end,:], mean_pow[HC_end+1:,:]]
[F_obs, clusters, cluster_p_values, H0] = mne.stats.permutation_cluster_test(perm_in, n_permutations=10000,
                                                                             n_jobs=1, out_type='mask')

# --- Data to longformat for Plotting ---
dat_plot_pow = []
for i in range(nSub-outSub):
    sub_tmp = np.ones([1, len(freqs)])*(i+1)
    if i < HC_end:
        group_tmp = np.ones([1, len(freqs)])
    else:
        group_tmp = np.ones([1, len(freqs)])*2
    dat_tmp = np.vstack([np.squeeze(sub_tmp), np.squeeze(
        group_tmp), np.squeeze(mean_pow[i,:]), freqs]).T
    dat_plot_pow.append(dat_tmp)

dat_plot_pow = np.array(dat_plot_pow)
dat_plot_pow = np.reshape(dat_plot_pow, [
                          dat_plot_pow.shape[0]*dat_plot_pow.shape[1], dat_plot_pow.shape[2]])
dat_plot_pow = pd.DataFrame(dat_plot_pow, columns=[
                            'SubID', 'Group', 'Pow', 'Freq'])

#%% Plotting

# --- Set Colors ---
Group_palette = get_colors()['PDvsHC_bin']


# ---- Set up Figure ---
fig = plt.figure(dpi=300, figsize=(8.4,4.2))
gs = plt.GridSpec(1, 1, height_ratios=[1], wspace=.35, hspace=.4)
ax = np.zeros(8, dtype=object)
ax[0] = fig.add_subplot(gs[0, 0])

# --- Average PowerSpectra with Cluster-based Permutation Test ---
sns.lineplot(data=dat_plot_pow, x="Freq", y="Pow", hue="Group",
             linewidth=3, ax=ax[0], palette=Group_palette)

# Make Axes pretty
ax[0].set_xlim([1, 45])
ax[0].set_xlabel('Frequency (Hz)',fontsize=16)
ax[0].set_ylabel('Power (a.u.)', fontsize=16,labelpad=15)
ax[0].tick_params(axis='both', which='major', labelsize=14) 

# Legend
ax[0].legend(frameon=False, fontsize=12)
leg = ax[0].axes.get_legend()
new_title = ''
leg.set_title(new_title)
new_labels = ['HC', 'PD']
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)

# Remove Box Around Subplot
sns.despine(ax=ax[0], top=True, right=True, left=False, bottom=False)

# Make y-labels pretty
ax[0].ticklabel_format(scilimits=(-1,10))

# Mark Significant Clusters
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax[0].axvspan(freqs[c.start], freqs[c.stop - 1],
                          color='r', alpha=0.3)
    

plt.savefig(plot_dir + 'Fig2_staticMotor_Pow.svg',transparent = True,bbox_inches="tight", 
            format='svg')