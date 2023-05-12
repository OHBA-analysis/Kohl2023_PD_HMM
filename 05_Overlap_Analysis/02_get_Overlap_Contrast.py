"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

Script plots Overlap measures between bursts and states

Script Loads Overlap Values between Beta Bursts and TDE-HMM States and 
assesses their significance.
To do this independent sample ttests are calculated between the overlap values
and null overlap values calculated from overlap between burst TC and rolled
HMM-State on-vs.-off time courses.

For each subject TDE-HMM state on-vs-off vector is shifted by another integer (randomely selected).
Directed ttest were calculated to focus on states that showed signifcantly larger
overlaps than null overlaps.

Output Data that is saved is input for Figure 5.

"""


import numpy as np
from scipy.stats import ttest_ind
import pickle
import random
import pandas as pd
from mne.stats import bonferroni_correction


#%% --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]#[1,2,3,4,5]
nPerm = 1000

proj_dir = '/path/to/proj_dir/'
outdir_plot = proj_dir + 'Results/StateBurst_Overlap/ds' + str(fs) + '/K' + str(K) + '/run' + str(nrepeate[0]) + '/'
outdir_dat = proj_dir + 'Data/Overlap/ds' + str(fs) + '/K' + str(K) +  '/run' + str(nrepeate[0]) + '/'
#data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'


indat = pickle.load( open( outdir_dat + "Overlap_Metrics.mat", "rb" ))
overlap_BurstNorm = indat['Overlap_BurstNorm']
null_overlap_BurstNorm = indat['Overlap_BurstNorm_Nulls']


#%% Permutation test of overlap with randome shift per participant - directed

# Creat Randome Numbers to select randome shift per subject
random.seed(1)
rShift = random.sample(range(nPerm),nSub) 

# Pick Null Distributions
null_burst = np.array([null_overlap_BurstNorm[iSub,rShift[iSub]] for iSub in range(nSub)])

# Calculate Ttests to see which states are significantly overlapping with bursts
t_burst, p_burst = ttest_ind(overlap_BurstNorm, null_burst, alternative='greater')

# Bonferroni correction
p_burst_bonf = bonferroni_correction(p_burst)


#%% Bring Data into shape and save for Plotting somewhere else

# ---- Set A few General Things ------

labels_all = ['State\n1', 'State\n2', 'State\n3', 'State\n4', 'State\n5', 'State\n6', 'State\n7', 'State\n8','State\n9', 'State\n10', 'State\n11', 'State\n12']
labels = labels_all[:K]

# Vectro marking where nulls will be
null_vect = np.ones(nSub*2)
null_vect[67:] = 0

sort_states = np.load('/path/to/Data/sortingVector/ClusterPowSorting.npy')


# --- Bring Data into Longformat for Seaborn Plotting ----


overlap_in = overlap_BurstNorm[:,sort_states]
null_in = null_burst[:,sort_states]
p = p_burst_bonf[1][sort_states]

df_freq = []
for i in range(len(labels)):
    over_tmp = np.hstack([overlap_in[:,i],null_in[:,i]]) 
    lab_tmp = np.ones([1, len(over_tmp)])*(i+1)#%[labels[i]]*df_in.shape[0]
    dat_tmp = np.vstack([np.squeeze(lab_tmp),np.squeeze(over_tmp),np.squeeze(null_vect)]).T
    df_freq.append(dat_tmp)

df_freq = np.vstack(df_freq)
df_freq = pd.DataFrame(df_freq, columns=['State','Overlap','Nulls'])


# --- Save Data for plotting elsewhere ---
mdict = {'df_freq': df_freq, 'p': p, 'labels': labels, 'overlap_in':overlap_in}
pickle.dump(mdict, open('/path/to/Data/V2_Plots/Overlap_Plot_Data.p', "wb"))