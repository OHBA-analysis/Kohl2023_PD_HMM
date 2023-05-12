#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:52:55 2022

@author: okohl

1) Script calculates network associated beta bursts (NABBs):

    Periods with overlap between occurence of a brain state and beta bursts
    from amplitude thresholding analysis are identified. Co-Occurences are
    stored in binary co-occurence vs no-co-occurence vector.
    From this binary co-occurence time course state metrics of NABBs are calculated.
    
    Importantly, NABB metrics and binary NABB on-vs.-off
    vectors are saved with this script!
    
2) Calculate Group Contrast between NAAB state metrics.
    Note: Beta Power is calculated separately with GLM-spectrum approach.
          Thus, this means it will be loaded in separately.
         
3) Vilone Plots are plotted to visualise the group contrast of the NAAB
    State Metrics. Violines of states that show decrease of beta power from
    mean when state onsets are depicted in shaded/muted colors.


"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pickle
import pandas as pd
import glmtools as glm
import seaborn as sns

import sys

sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")

from plotting import p_for_annotation, split_barplot_annotate_brackets, get_colors
from burst_analysis import burst_detection, burst_time_metrics, custom_burst_metric
    

#%% Get State Burst Overlap and Metrics


# --- Set Parameters and Dirs ----
# Parameter
nSub = 67
K = 8
fs = 250
nrepeate = [7]#[1,2,3,4,5]

proj_dir = '/path/to/proh_dir/'
outdir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/'
data_dir = proj_dir + 'Data/spectra_toPython/ds' + str(fs) + '/K' + str(K) + '/'
data_outdir = proj_dir + 'Data/Burst_x_State_Metrics/ds250/K' + str(K) + '/'


# ---- Start Loop  through outputs of different HMM runs
for irun in nrepeate: 
    
    # --- Start Loop through Participants to save Copes and Tstats ---   
    burst_overlap = np.zeros([nSub])
    burst_FOs = np.zeros([nSub,K,1])
    burst_rates = np.zeros([nSub,K,1])
    burst_meanLTs = np.zeros([nSub,K,1])
    burst_meanITs = np.zeros([nSub,K,1])
    
    for iSub in range(nSub):
        
        print('Loading Data for Subject' + str(iSub+1))
        
        # --- Load Data ---
        # Load Data
        file = 'run' + str(irun) + '/Subject' + str(iSub + 1) + '_HMMout.mat'   
        data_in = loadmat(data_dir + file)
        XX_r = data_in['subj_data'][:,17] # Just Select Parcel 17 or 18
        XX_l = data_in['subj_data'][:,18]
        Gamma = data_in['subj_Gamma']
        
        # Binarise state probabilities
        Gamma_bin = np.array([Gamma[:,kk] > .75 for kk in range(K)]).astype(float)
        
        
        # --- Extract Beta Bursts ----
        
        # Get Burst on off vector and normalised data
        freq_range = [18, 25]
        fsample = 250

        is_burst_l, norm_data_l = burst_detection(XX_l, freq_range, fsample = fsample, normalise = 'none', 
                                              threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
        
        is_burst_r, norm_data_r = burst_detection(XX_r, freq_range, fsample = fsample, normalise = 'none', 
                                              threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)

        
        is_burst = np.logical_or(is_burst_l,is_burst_r)
        norm_data = np.mean([np.squeeze(norm_data_l), np.squeeze(norm_data_r)],axis = 0)
        
        
        # How do Bursts in left and right parcel overlap ?
        burst_overlap[iSub] = sum(is_burst_l == is_burst_r)/len(is_burst_l)
        print(str(np.round(burst_overlap[iSub],2)*100) + '% Overlap between right hemisphere and left hemisphere Bursts.')
        
        
        # --- Combine Burst and State Vector ---
        
        burst_amps = []
        burst_lifetimes = []
        is_BurstState_out = np.zeros(Gamma_bin.shape)
        
        # ---- Calculate overlap between Bursts and States -----
        for iState in range(Gamma_bin.shape[0]):
            
            is_BurstState = np.logical_and(Gamma_bin[iState,:],is_burst)
                  
            # ----- Calculate Burst Metrics -----
            
            # Get Burst Time Metrics
            burst_time_dict = burst_time_metrics(is_BurstState, fsample)
                
            # Get Average Measures
            burst_FOs[iSub,iState] = burst_time_dict['Burst Occupancy']
            burst_rates[iSub,iState] = burst_time_dict['Burst Rate']
            burst_meanLTs[iSub,iState] = np.nanmean(burst_time_dict['Life Times'])
            burst_meanITs[iSub,iState] = np.nanmean(burst_time_dict['Interval Times'])
            
            is_BurstState_out[iState,:] = is_BurstState
            
        # Save binary vector indicating State_x_Burst overlap // different betas
        np.save(data_outdir + '/run' + str(irun) + '/is_StateBurst/is_StateBurst_Subject' + str(iSub), is_BurstState_out) 
        
    # Prepare dat for saving
    MetricLabels = ['Fractional Occupancy', 'Mean Lifetimes', 'Mean Interval Times','Burst Rates']
    allMetrics = [burst_FOs, burst_meanLTs, burst_meanITs, burst_rates]     
    
    # ------ Results for further Plotting ------
    mdict = {'metrics': allMetrics, 'metric labels': MetricLabels, 'burst_overlap': burst_overlap}
    pickle.dump(mdict, open(outdir + 'run' + str(irun) + '/Burst_x_State_Metrics.mat', "wb"))



#%% --- Creat second StateBurst Metric File with Burst Amplitudes from GLM-Spectrum ---

fs = 250
K = 8
irun = 7

# Set dirs
proj_dir = '/path/to/proj_dir/'
indir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
spectrum_dir = proj_dir + 'Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'

# ---- Load StateBurst Metrics -----
allMetrics = pickle.load(open(indir + 'Burst_x_State_Metrics.mat', "rb"))['metrics']
MetricLabels = ['Fractional Occupancy', 'Mean Lifetimes', 'Mean Interval Times','Burst Rates','Mean Beta Power']

# ---- Load GLM-Spectrum Output ----
beta_spectrum = loadmat(spectrum_dir + 'StateBurst_GLM_BetaPower.mat')['out'][:,:,np.newaxis]

# ---- Add GLM Spectrum results -----
allMetrics[4] = beta_spectrum

# ---- Save Data ----
mdict = {'metrics': allMetrics, 'metric labels': MetricLabels}
pickle.dump(mdict, open(outdir + 'run' + str(irun) + '/Burst_x_State_Metrics_GLM_Pow.mat', "wb"))



#%% --- Calculate NABB Metric Group Contrasts ---


# ------------------
# --- load data ----
# ------------------

# ---- Load Confounds ----
behav_in = '/path/to/behavioral_data/'
df_in = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')

# Remove Rows with nan and store indices of good Rows
df_in = df_in[['Group','Handedness','Gender','Age','Education']]
in_ind = np.prod(df_in.notna().values,axis = 1)
in_ind = np.ma.make_mask(in_ind)
df_in = df_in[in_ind] 


# -------------------------------
# --- Start Loop across HMMs ----
# -------------------------------


#  Set Parameters  
fs = 250
number_of_states = [8]
nrepeate = [7]#[1,2,3,4,5]

for K in number_of_states: # Loop over different states
    for irun in nrepeate:  # Loop over different HMM inferences

        # Set dirs
        proj_dir = '/path/to/[rpj_dir'
        indir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        outdir_plot = proj_dir + 'Results/State_x_Burst_Metric_GroupContrast/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        data_outdir = proj_dir + 'Data/Burst_x_State_Metrics_GroupContrast/ds250/K' + str(K) + '/run' + str(irun) + '/'

        # ---- Load Burst Metrics -----
        allMetrics = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metrics']
        MetricLabels = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metric labels']

        # ------------------------------------------------------------
        # --- GLM comparing metrics while controlling for confounds ---
        # ------------------------------------------------------------
             
        p_all = []
        tstat_all = []
        thresh_all = []
        
        for iMetric in range(0,len(allMetrics)):
            data = glm.data.TrialGLMData(data = np.squeeze(allMetrics[iMetric][in_ind]),
                                         category_list=df_in['Group'].values,
                                         covariate = df_in['Age'].values,
                                         gender = df_in['Gender'].values,
                                         handedness = df_in['Handedness'].values,
                                         education = df_in['Education'].values,
                                         num_observations=allMetrics[3][in_ind].shape[0] )
            
            # ----- Specify regressors and Contrasts in GLM Model -----
            DC = glm.design.DesignConfig()
            DC.add_regressor(name='HC',rtype='Categorical',codes=1)
            DC.add_regressor(name='PD',rtype='Categorical',codes=2)
            DC.add_regressor(name='Gender', rtype='Parametric', datainfo='gender', preproc='z')
            DC.add_regressor(name='Handedness', rtype='Parametric', datainfo='handedness', preproc='z')
            DC.add_regressor(name='Education', rtype='Parametric', datainfo='education', preproc='z')
            DC.add_regressor(name='Age', rtype='Parametric', datainfo='covariate', preproc='z')
             
            DC.add_simple_contrasts()
            DC.add_contrast(name='HC < PD', values=[-1, 1, 0, 0, 0, 0])
            
            #  ---- Create design martix ----
            des = DC.design_from_datainfo(data.info)
             
            # ---- fit GLM -----
            model = glm.fit.OLSModel(des,data)
            
            
            # -------------------------------------
            # Permutation Test Pooling Across States
            # ---------------------------------------
            
            contrast = 6 # select Group Contrast
            metric_perm = 'tstats' # add the t-stats to the null rather than the copes
            nperms = 10000  # for a real analysis 1000+ is best but 100 is a reasonable approximation.
            nprocesses = 4 # number of parallel processing cores to use
            
            # calculate GLM
            P = glm.permutations.MaxStatPermutation(des, data, contrast, nperms,
                                                    metric=metric_perm, nprocesses=nprocesses,
                                                    pooled_dims=(1))
            
            # Creat Vars for further Plotting:
            tstat = model.tstats[contrast]
            thresh = P.get_thresh([95, 99])
            sig05 = abs(tstat) > thresh[0] # Mask of tstats < .05
            sig01 = abs(tstat) > thresh[1]
            
            # Get p - two sided/undirected
            nulls = P.nulls
            p = [(abs(tstat[t])<abs(nulls)).sum()/nulls.size for t in range(tstat.size)]
            
            print('p =' + str(p))
        
            tstat_all.append(tstat)
            thresh_all.append(thresh)
            p_all.append(p)
            
            # ------ Results for further Plotting ------
            mdict = {'tstats': tstat, 'thresh': thresh, 'sig05': sig05,
                     'sig01': sig01, 'p': p, 'model': model, 'P': P, 
                     'Metrics': MetricLabels[iMetric]}
            pickle.dump(mdict, open(data_outdir + MetricLabels[iMetric] +
                                    "_StateBurst_GroupComp_GLM.mat", "wb"))
          

# ---------------------------------------------------
# %% Make Plots for State Metric Group Comparison GLMs
# ---------------------------------------------------
# Script creates split Violin Plots to depict group 
# differences in State Metrics.
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
labels_metric = allMetrics

# For Plotting
labels = ['State \n 1', 'State \n 2', 'State \n 3', 'State \n 4', 'State \n 5', 'State \n 6',
          'State \n 7', 'State \n 8', 'State \n 9', 'State \n 10', 'State \n 11', 'State \n 12']
labels_plot = ['Fractional Occupancy', 'Mean Lifetimes',
               'Mean Interval Times', 'State Rates', 'Motor Beta Power']
 
# Load State Sorting vector
sort_states = np.load('/path/to/Data/sortingVector/ClusterPowSorting.npy')

# --- Start Loop across HMMs ---


# Set Params
fs = 250
number_of_states = [8]
nrepeate = [7] # run with lowest free Energy

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
        
        # --- Set up Grid for Plotting ---
        x = 1.2
        fig = plt.figure(dpi=300, figsize=(12.8*x, 9.6*x))
        gs = plt.GridSpec(2, 3, height_ratios=[1, 1], wspace=.3, hspace=.25)
        ax = np.zeros(5, dtype=object)
        ax[0] = fig.add_subplot(gs[0, 0])
        ax[1] = fig.add_subplot(gs[0, 1])
        ax[2] = fig.add_subplot(gs[0, 2])
        ax[3] = fig.add_subplot(gs[1, 0])
        ax[4] = fig.add_subplot(gs[1, 1])
        
        for iMetric, metric in enumerate(labels_metric): # -1 to exclude switch rates
            
            # --- Store vars from GLM ---
            GLM_in = pickle.load(open(data_outdir + metric + 
                                      "_StateBurst_GroupComp_GLM.mat", "rb"))
            tstats = GLM_in['tstats'][sort_states]
            thresh = GLM_in['thresh']
            p = [GLM_in['p'][i] for i in sort_states]
            
            df_states = pd.DataFrame(np.squeeze(allMetrics[iMetric][:,sort_states]), columns = labels[:K]) # -1 because of Python indexing
            
             
            # Bring State Metric Data in longformat
            df_fact = []
            for i, label in enumerate(labels[:K]):
                group_tmp = group_vect
                lab_tmp = np.ones([1, len(df_states)])*(i+1)
                cl_tmp = df_states[label]
                dat_tmp = np.vstack([np.squeeze(lab_tmp), group_tmp,np.squeeze(cl_tmp)]).T
                df_fact.append(dat_tmp)
            
            df_fact = np.array(df_fact)
            df_fact = np.reshape(df_fact, [
                                      df_fact.shape[0]*df_fact.shape[1], df_fact.shape[2]])
            df_fact = pd.DataFrame(df_fact, columns=[
                                        'State', 'Group',metric])
           
            # ---  Violine Plot ---
            sns.violinplot(x="State", y=metric, hue="Group",inner = "quartile",
                                data=df_fact, palette=PDvsHC_palette, 
                                cut=.5 , bw=.35 ,split=True,ax = ax[iMetric])
            
            # Shading of Last 4 Violines
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
                l.set_linewidth(1.2)
                l.set_color('#AEAEAE')
                l.set_alpha(0.8)
            
            # Labels and Ticks
            if iMetric > 1:
                ax[iMetric].set_xlabel('State', fontsize=16)
            else:
                ax[iMetric].set_xlabel('')
                
            ax[iMetric].set_xticklabels(np.arange(1,K+1), fontsize=16) #becasue of np.arrang does not include end of specified range
            ax[iMetric].set_ylabel(labels_plot[iMetric])
            ax[iMetric].yaxis.label.set_size(16)
            ax[iMetric].tick_params(axis='both', which='major', labelsize=14)
            ax[iMetric].tick_params(axis='both', which='minor', labelsize=14)
            
            # Legend
            if iMetric == 2:
                ax[iMetric].legend(prop = {'size':16}, frameon=False)
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
            # get signifiance levels for different states!     
            p_plt = []      
            for iState in range(len(labels[:K])):
                if abs(tstats[iState]) > thresh[1]:
                    p_plt.append(p_for_annotation(.003))
                elif abs(tstats[iState]) > thresh[0]:
                    p_plt.append(p_for_annotation(.03))
                else:
                    p_plt.append(p_for_annotation(.1))
            
            # Parameters for significance star plottinng
            heights = np.max(allMetrics[iMetric][:,sort_states],axis = 0) # Get height of violine plots
            bars = np.arange(len(heights)).astype(int) # Get x Position of violines
            ax_in = ax[iMetric] # define axis to which sig stars are plotted
          
            # Plot Starts in Violine Plot    
            for ii in range(len(p_plt)):
                if metric == 'Mean Interval Times':
                    barh = .05
                    df = .008
                elif metric == 'Mean Lifetimes':
                    barh = .0015
                    dh = .001
                elif metric == 'Fractional Occupancy':
                    barh = .0012
                    dh = .008
                elif metric == 'Mean Amplitude':
                    barh = .02
                    dh = .0075
                else:
                    barh = .08
                    dh = .008
                    
                split_barplot_annotate_brackets(bars[ii], bars[ii], np.array(p_plt)[ii], bars, heights, ax_in=ax_in,
                                          yerr=None, dh=dh, barh=barh, fs=20, maxasterix=None)
        
        # Save Plot    
        plt.savefig(outdir_plot + 'State_x_Burst_Metric_GroupComp_ViolinePlots_wAmp.svg',
                       transparent=True, bbox_inches="tight",format='svg')
            