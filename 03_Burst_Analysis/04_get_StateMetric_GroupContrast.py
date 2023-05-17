#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:14:14 2022

@author: okohl

Scripts calculates State Metric Group Contrasts with GLMs while controling for confounds.
Findings are plotted afterwards with violine plots. -> Figure 2

# Associations between burst metrics and UPDRS scores are also assessed with
# GLMs controlling confound and plotted.

"""

import pickle
import numpy as np
import pandas as pd
import glmtools as glm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("/path/to/helpers/")
from plotting import p_for_annotation, split_barplot_annotate_brackets, get_colors

# Set Plot Dir
plot_dir = '/path/to/Results/BurstMetric_Comparisons/ds250/K8/run7/'


#%% load data

# ---- Burst Metrics -----
outdir = '/path/to/Data/BurstMetrics/ds250/K8/run7/'
metrics = pickle.load(open(outdir + 'BurstMetrics_GLMBeta.mat', "rb"))['metrics']

# ---- Load Confounds ----
behav_in = '/path/to/TDE_HMM_project/data/behavioral_data/'
df_in = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')

# Remove Rows with nan and store indices of good Rows
df_in = df_in[['Group','Handedness','Gender','Age','Education']]
in_ind = np.prod(df_in.notna().values,axis = 1)
in_ind = np.ma.make_mask(in_ind)
df_in = df_in[in_ind] 

#%% GLM comparing Burst Metrics while controlling for confounds
p_all = []
tstat_all = []
thresh_all = []

# Loop Through State Metrics
for iMetric in range(0,4):
    
    print(str(iMetric))
    
    # ----- Set up Data Object ----
    data = glm.data.TrialGLMData(data=metrics[iMetric][in_ind],
                                 category_list=df_in['Group'].values,
                                 covariate = df_in['Age'].values,
                                 gender = df_in['Gender'].values,
                                 handedness = df_in['Handedness'].values,
                                 education = df_in['Education'].values,
                                 num_observations=metrics[3][in_ind].shape[0] )
    
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
                                            pooled_dims=())
    
    # Creat Vars for further Plotting:
    tstat = model.tstats[contrast]
    thresh = P.get_thresh([95, 99])
    sig05 = abs(tstat) > thresh[0] # Mask of tstats < .05
    sig01 = abs(tstat) > thresh[1]
    
    # Get p - two sided/undirected
    nulls = P.nulls
    p = [(abs(tstat)<abs(nulls)).sum()/nulls.size for t in range(tstat.size)]
    
    print(str(iMetric) + ': t = ' + str(tstat) + '; p =' + str(p))

    tstat_all.append(tstat)
    thresh_all.append(thresh)
    p_all.append(p)

#%% Plot Group Contrast Violines


# ---------------------------------------------------
#  Make Plots for Burst Metric Group Comparison GLMs
# ---------------------------------------------------
# Script creates split Violin Plots to depict group 
# differences in Burst Metrics.
#
# Significant Differences are marked with ** or *
# ----------------------------------------------------
  
              
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

# --- Set Colors ---
PDvsHC_palette = get_colors()['PDvsHC_palette']


# --- Define Labels ---

# For Plotting
labels = ['Burst']
labels_plot = ['Fractional Occupancy', 'Mean Lifetimes (sec)',
               'Burst Rates','Beta Power (a.u.)']

# -- Set up Grid for Plotting --
x = .8
fig = plt.figure(dpi=300, figsize=(12.8*x, 5*x))
gs = plt.GridSpec(1, 4, wspace=.7, hspace=.1)
ax = np.zeros(4, dtype=object)
ax[0] = fig.add_subplot(gs[0, 0])
ax[1] = fig.add_subplot(gs[0, 1])
ax[2] = fig.add_subplot(gs[0, 2])
ax[3] = fig.add_subplot(gs[0, 3])

# Loop Through Metrics
for iMetric, metric in enumerate(labels_plot):
    
    # Load important vars from GLM
    tstats = tstat_all[iMetric][0]
    thresh = thresh_all[iMetric]
    metric_in = metrics[iMetric][in_ind]
            
    # Bring State Metric Data in longformat
    df_fact = []
    group_tmp = group_vect[in_ind]
    lab_tmp = np.ones([1, len(metric_in)])
    cl_tmp = metric_in
    dat_tmp = np.vstack([np.squeeze(lab_tmp), group_tmp,np.squeeze(cl_tmp)]).T
    
    df_fact = pd.DataFrame(dat_tmp, columns=[
                                'Burst', 'Group',metric])
       
    
    # --- Start plotting of Violine Plots ---    
    sns.violinplot(x="Burst", y=metric, hue="Group",inner = "quartile",
                        data=df_fact, palette=PDvsHC_palette, 
                        cut=.5 , bw=.35 ,split=True,ax = ax[iMetric])
    
    # Set Line Colors
    for l in ax[iMetric].lines:
        l.set_linestyle('--')
        l.set_linewidth(.8)
        l.set_color('#AEAEAE')
        l.set_alpha(0.8)
        
    for l in ax[iMetric].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(2.5)
        l.set_color('#AEAEAE')
        l.set_alpha(0.8)
    
    # Labels and Ticks       
    ax[iMetric].set_xlabel('', fontsize=16)
     
    ax[iMetric].set_xticklabels('', fontsize=16)
    ax[iMetric].set_ylabel(labels_plot[iMetric])
    ax[iMetric].yaxis.label.set_size(16)
    ax[iMetric].tick_params(axis='y', which='major', labelsize=14)
    ax[iMetric].tick_params(axis='y', which='minor', labelsize=14)
    ax[iMetric].tick_params(axis='x', which='both', bottom = False,
                            top = False, labelbottom = False)
    
    # Set Legend
    if iMetric == 3:
        ax[iMetric].legend(prop = {'size':16}, frameon=False, loc='upper right', bbox_to_anchor=(1.5, 1))
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
    
    
    # --- Get Significance stars in violine plots ---
    
    # get signifiance levels for different states!     
    p_plt = []
    if abs(tstats) > thresh[1]:
        p_plt.append(p_for_annotation(.003))
    elif abs(tstats) > thresh[0]:
        p_plt.append(p_for_annotation(.03))
    else:
        p_plt.append(p_for_annotation(.1))
    
    # Parameters for significance star plottinng
    heights = np.max(metrics[iMetric],axis = 0) # Get height of violine plots
    bars = np.arange(len(heights)).astype(int) # Get x Position of violines
    ax_in = ax[iMetric] # define axis to which sig stars are plotted
      
    # Plot Starts in Violine Plot    
    split_barplot_annotate_brackets(bars, bars, np.array(p_plt), bars, heights, ax_in=ax_in,
                                  yerr=None, dh=.005, barh=0.015, fs=20, maxasterix=None)
    
    # Save Plot    
    plt.savefig(plot_dir + 'BurstMetric_GroupComp_ViolinePlots_V2.svg',transparent = False,
                bbox_inches="tight",format='svg')
        



# #%% --- GLMs assessing Burst Amplitude UPDRS associations ---


# # --- Load UPDRS scores and confounds

# # --- Load UPDRS scores ---
# behav_in = '/path/to//behavioral_data/'
# behav = pd.read_pickle(behav_in + 'sourceBehave_PDfactors.pkl')

# # Remove Rows with nan and store indices of good Rows
# df_PD = behav[['Age','Education','UPDRS-Motor',
#             'Midline Function', 'Rest Tremor', 'Rigidity','Bradykinesia Right Upper Extremity',
#             'Bradykinesia Left Upper Extremity','Postural & Kinetic Tremor',
#             'Lower Limb Bradykinesia','Years Since Diagnosis', 'Lateralisation Score']]
# in_ind_PD = np.prod(df_PD.notna().values,axis = 1)
# in_ind_PD = np.ma.make_mask(in_ind_PD)
# df_in_PD = df_PD[in_ind_PD]

# df_in_PD['Bradykinesia/Rigidity'] = df_in_PD['Bradykinesia Right Upper Extremity'] + df_in_PD['Bradykinesia Left Upper Extremity'] + df_in_PD['Lower Limb Bradykinesia'] + df_in_PD['Rigidity']
# df_in_PD['Tremor'] = df_in_PD['Rest Tremor'] + df_in_PD['Postural & Kinetic Tremor']

# labels_PD = ['Bradykinesia/Rigidity','Tremor']


# # --- load behavioral data put the in dictionary of covariates ---
# behav_in = '/path/to/behavioral_data/'
# df = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')

# # Remove Rows with nan and store indices of good Rows
# df_in = df[['Group','Handedness','Gender','Age','Education']]
# in_ind = np.prod(df_in.notna().values,axis = 1)
# in_ind = np.ma.make_mask(in_ind)
# df_in = df_in[in_ind_PD]


# # --- Calculate GLMs ---

# # Define Data Object  for GLM
# data = glm.data.TrialGLMData(data= metrics[3][in_ind_PD],
#                              Brad_Rigidity=df_in_PD[labels_PD[0]],
#                              Tremor=df_in_PD[labels_PD[1]],
#                              covariate=df_in['Age'].values,
#                              gender=df_in['Gender'].values,
#                              handedness=df_in['Handedness'].values,
#                              education=df_in['Education'].values,
#                              num_observations=metrics[3][in_ind_PD].shape[0])


# # Define Model for GLM
# DC = glm.design.DesignConfig()
# DC.add_regressor(name='Constant',rtype='Constant')
# DC.add_regressor(name='Gender', rtype='Parametric',
#                   datainfo='gender', preproc='z')
# DC.add_regressor(name='Handedness', rtype='Parametric',
#                   datainfo='handedness', preproc='z')
# DC.add_regressor(name='Education', rtype='Parametric',
#                   datainfo='education', preproc='z')
# DC.add_regressor(name='Age', rtype='Parametric',
#                   datainfo='covariate', preproc='z')
# DC.add_regressor(name='Brad_Rigidity', rtype='Parametric',
#                  datainfo='Brad_Rigidity', preproc='z')
# DC.add_regressor(name='Tremor', rtype='Parametric',
#           datainfo='Tremor', preproc='z')

# # Add Contrast   
# DC.add_simple_contrasts()

# # Create design martix
# des = DC.design_from_datainfo(data.info)

# # --- fit the model ---
# model = glm.fit.OLSModel(des,data)


# # --- Test for significance with Permutation tests ---
# p_all_cor = []
# sig_cor = []
# tstats_all = []
# P_all = []
# p_sig_all = []
# for iContrast in range(5,7):

#     # --- Permutation tests pooling across States ---
    
#     contrast = iContrast # select the UPDRS contrast
#     metric = 'tstats' # add the t-stats to the null rather than the copes
#     nperms = 10000  # for a real analysis 1000+ is best but 100 is a reasonable approximation.
#     nprocesses = 4 # number of parallel processing cores to use
#     pooled_dims = ()
    
    
#     P = glm.permutations.MaxStatPermutation(des, data, contrast, nperms,
#                                             metric=metric, nprocesses=nprocesses,
#                                             pooled_dims = pooled_dims)
    
#     P_all.append(P)


#     # Bring outout into [state metric][State] shape
#     thresh = P.get_thresh([95, 99]) 
#     tstat = model.tstats[iContrast]
    
#     p_all_cor.append(thresh)                                   
#     sig_cor.append(abs(model.tstats[iContrast]) > thresh[0])
#     tstats_all.append(tstat)
#     p_sig_all.append((abs(tstat)<abs(P.nulls)).sum()/P.nulls.size)
    

# #%% --- Plot Bradykinesia/Rigidity x Burst amplitude association ---

# # ---- Set Up ------
# fig = plt.figure(dpi=300)
# gs = plt.GridSpec(1, 1, wspace=.6, hspace=.32)
# ax = np.zeros(1, dtype=object)
# ax = fig.add_subplot()

# # ------  Do Regression Plots ----------
# dat_plot = {'Burst Amplitude': np.squeeze(metrics[3][in_ind_PD]),
#             'Bradykinesia/Rigidity' : df_in_PD['Bradykinesia/Rigidity']}
# dat_plot = pd.DataFrame(dat_plot)

# sns.set_style("ticks")
# sns.regplot(x="Bradykinesia/Rigidity", y="Burst Amplitude", data=dat_plot, ax = ax, color = PDvsHC_palette[1] )

# ax.set_ylabel('Beta Power (a.u.)', fontsize = 12, labelpad = 10)
# ax.set_xlabel('Bradykinesia/\nRigidity', fontsize = 12, labelpad = 5)

# ax.ticklabel_format(scilimits=(-1,1))
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)

# ax.text(.95,.98,'T-Stat = ' + str(np.round(tstats_all[0],2)[0]) + ', p = ' + str(np.round(p_sig_all[0],2)),
#         horizontalalignment='right',
#         verticalalignment='top',
#         transform = ax.transAxes,
#         fontsize = 14)

# # Remove Box Around Subplot
# sns.despine(ax=ax, top=True, right=True, left=False,
#         bottom=False, offset=None, trim=False)

# # ----- Save Figure ------
# plt.savefig(plot_dir + 'BurstAmplitude_UPDRS_GLM_corr.svg',
#                transparent=True, bbox_inches="tight",format='svg')

