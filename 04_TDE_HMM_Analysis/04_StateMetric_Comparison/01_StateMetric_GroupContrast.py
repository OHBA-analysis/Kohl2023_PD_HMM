#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 10:52:30 2022

@author: okohl

1)  Script Calculates State Metric Group Comparisons between HC and PD patients
    with GLMs controlling for confounds. 
    
2)  Significance calculatde with MaxTstatistic Permutation tests pooling 
    across States!

3)  Make Overview visualising Group Contrasts of all States for all State Metrics
    in Violine plots. -> SI 5
    
One can run script accross all HMM fits by putting all run numbers as runs.
Outputs are what will be loaded for robustness analysis
"""

import glmtools as glm
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection

import sys
sys.path.append("/home/okohl/Documents/HMM_PD_V07/Scripts/helpers/")
from plotting import p_for_annotation, split_barplot_annotate_brackets, get_colors


# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [1,2,3,4,5,6,7,8,9,10]
sampling_rates = [250]
number_of_states = [8,10,12]
embeddings = [7]
for irun in runs:
    for fsample in sampling_rates:
        for emb in embeddings:
            for K in number_of_states:
                
                print('Running Analysis for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
      
                # ----------------------------------------------------
                #%% GLMs Calculating HMM State Metric Group Differences
                # -----------------------------------------------------
                # Script uses same GLM-Models as static PowSpec Group Comparison
                # 1) GLMs are calculated
                # 2) Permutation Tests assessing significance of 
                #    predictors are calculated
                # ----------------------------------------------------
                
                proj_dir = '/path/to/proj_dir'
                dat_dir = proj_dir + 'Data/StateMetrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                outdir_plot = proj_dir + 'Results/StateMetric_GroupContrast/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                # ---- Load Confounds and store them in df_in -----
                behav_in = '/path/to/behavioral_data/'
                df_in = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')
                
                # Remove Rows with nan and store indices of good Rows
                df_in = df_in[['Group','Handedness','Gender','Age','Education']]
                #df_in = df_in.drop([15]) # Drop Outlier Subject 
                in_ind = np.prod(df_in.notna().values,axis = 1)
                in_ind = np.ma.make_mask(in_ind)
                df_in = df_in[in_ind]
                
                
                # ---- Start Loop calculating Groupd Differences for different labels -----
                
                # ---- Specify labels for loop
                labels_metric = ['fractional_occupancy','mean_lifetimes', 'mean_intervaltimes','state_rates', 'State_GLM_BetaPower']
                    
                #% Start Loop Calculating GLM for each State Metric
                for iMetric, metric in enumerate(labels_metric):
                    
                    print('running GLM for ' + str(metric))
                    
                    in_data = loadmat(dat_dir + metric + ".mat")['out']
                    in_data = in_data[:,np.newaxis,:]
                    in_data = in_data[in_ind,:,:]
                    
                    # --- Define Dataset for GLM -----    
                    data = glm.data.TrialGLMData(data=in_data,
                                                 category_list=df_in['Group'].values,
                                                 covariate = df_in['Age'].values,
                                                 gender = df_in['Gender'].values,
                                                 handedness = df_in['Handedness'].values,
                                                 education = df_in['Education'].values,
                                                 num_observations=in_data.shape[0] )
                    
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
                    des.plot_summary(savepath=outdir_plot + '/Model_Checks/' + metric + '_summary')
                    des.plot_efficiency(savepath=outdir_plot + '/Model_Checks/' + metric + '_efficiency')
                     
                    # ---- fit GLM -----
                    model = glm.fit.OLSModel(des,data)
                    
                    
                    # -------------------------------------
                    # Permutation Test Pooling Across States
                    # ---------------------------------------
                    
                    contrast = 6 # select Group Contrast
                    metric_perm = 'tstats' # add the t-stats to the null rather than the copes
                    nperms = 10000  # for a real analysis 1000+ is best but 100 is a reasonable approximation.
                    nprocesses = 4 # number of parallel processing cores to use
                    
                    # Pool max t-stats across second dimension. Corrects for multiple comparisons across States.
                    perm_args = {'pooled_dims': (2)}
                    
                    # calculate GLM
                    P = glm.permutations.MaxStatPermutation(des, data, contrast, nperms,
                                                            metric=metric_perm, nprocesses=nprocesses,
                                                            pooled_dims=(2))
                    
                    # Creat Vars for further Plotting:
                    tstats= model.tstats[contrast,:,:]
                    thresh = P.get_thresh([95, 99])
                    sig_mask = abs(tstats) > thresh[0] # Mask of tstats < .05
                    
                    # Get p - two sided/undirected
                    nulls = P.nulls
                    p = [(abs(tstats[:,t])<abs(nulls)).sum()/nulls.size for t in range(tstats.size)]
                    
                    # ------ Results for further Plotting ------
                    mdict = {'tstats': tstats, 'thresh': thresh, 'sig_mask': sig_mask,
                             'model': model, 'P': P, 'Metrics': labels_metric[iMetric],
                             'p': p}
                    pickle.dump(mdict, open(dat_dir + labels_metric[iMetric] +
                                            "_GroupComp_GLM.mat", "wb"))
                
             
                
                
# ---------------------------------------------------
# %% Make Plots for State Metric Group Comparison GLMs
# ---------------------------------------------------
# Script creates split Violin Plots to depict group 
# differences in State Metrics.
#
# Significant Differences are marked with ** or *
# ----------------------------------------------------


# ------------------------------
# colors
col_dict = get_colors()

PDvsHC_palette = col_dict['PDvsHC_palette']

#------------------------------


# Set Dirs
irun = 7 # HMM with lowest FE when fitting 8 States
HMM_indir = proj_dir + '/Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

GLM_indir = proj_dir + '/Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                

# --- Get Participant vector ---
subIDs = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19',
                 '51','52','53','54','56','57','58','59','61','62','63','64','65','66','67','68','69','70','71',
                 '101','102','103','104','105','106','107','109','110','116','117','118',
                 '151','153','154','155','156','157','159','160','161','162','163','164','165','166','167','168','169','170']

HCsub = np.array(range(0, subIDs.index('71')+1))
PDsub = np.array(range(HCsub[-1]+1, subIDs.index('170')+1))
allSub = np.hstack([HCsub, PDsub])
iMetric = 0
group_vect = np.ones_like(allSub)
group_vect[allSub > len(HCsub)-1] = 2


# --- Define Labels ---
# To Load State Metrics
labels_metric = ['fractional_occupancy','mean_lifetimes', 'mean_intervaltimes','state_rates', 'State_GLM_BetaPower']

# For Plotting
labels = ['State \n 1', 'State \n 2', 'State \n 3', 'State \n 4', 'State \n 5', 'State \n 6',
          'State \n 7', 'State \n 8', 'State \n 9', 'State \n 10', 'State \n 11', 'State \n 12']
labels_plot = ['Fractional Occupancy', 'Mean Lifetimes (sec)',
               'Mean Interval Times (sec)', 'State Rates', 'Motor Beta Power (a.u.)']

# Load State Sorting vector
sort_states = np.load(proj_dir + 'Data/sortingVector/ClusterPowSorting.npy')


# --- Set up Grid for Plotting ---
x = 1.2
fig = plt.figure(dpi=300, figsize=(12.8*x, 9.6*x))
gs = plt.GridSpec(2, 3, height_ratios=[1, 1], wspace=.4, hspace=.25)
ax = np.zeros(5, dtype=object)
ax[0] = fig.add_subplot(gs[0, 0])
ax[1] = fig.add_subplot(gs[0, 1])
ax[2] = fig.add_subplot(gs[0, 2])
ax[3] = fig.add_subplot(gs[1, 0])
ax[4] = fig.add_subplot(gs[1, 1])

for iMetric, metric in enumerate(labels_metric): # -1 to exclude switch rates
    
    # Load importantb vars from GLM
    GLM_in = pickle.load(open( GLM_indir + metric + "_GroupComp_GLM.mat", "rb" ) ) # adjust
    tstats = GLM_in['tstats'][:,sort_states]
    thresh = GLM_in['thresh']
    
    # Load State Metrics inferred with HMM-MAR Toolbox
    in_data = loadmat(HMM_indir + metric + ".mat")['out'][:,sort_states]
    df_states = pd.DataFrame(in_data, columns = labels[:K]) # -1 because of Python indexing
    
     
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
    
    if iMetric == 2:
        #Legend
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
        if abs(tstats[:,iState]) > thresh[1]:
            p_plt.append(p_for_annotation(.003))
        elif abs(tstats[:,iState]) > thresh[0]:
            p_plt.append(p_for_annotation(.03))
        else:
            p_plt.append(p_for_annotation(.1))
    
    # Parameters for significance star plottinng
    heights = np.max(in_data,axis = 0) # Get height of violine plots
    bars = np.arange(len(heights)).astype(int) # Get x Position of violines
    ax_in = ax[iMetric] # define axis to which sig stars are plotted
  
    # Plot Starts in Violine Plot    
    for ii in range(len(p_plt)):
        if metric == 'mean_intervaltimes':
            barh = .17
        elif metric == 'mean_lifetimes':
            barh = .000000001
        elif metric == 'fractional_occupancy':
            barh = .01
        elif metric == 'State_GLM_BetaPower':
            barh = .02
        else:
            barh = .08
            
        split_barplot_annotate_brackets(bars[ii], bars[ii], np.array(p_plt)[ii], bars, heights, ax_in=ax_in,
                                  yerr=None, dh=.008, barh=barh, fs=20, maxasterix=None)

# Save Plot    
plt.savefig(outdir_plot + 'StateMetric_GroupComp_ViolinePlots_wAmp.svg',
            transparent = False, format='svg', bbox_inches = 'tight' )
    

