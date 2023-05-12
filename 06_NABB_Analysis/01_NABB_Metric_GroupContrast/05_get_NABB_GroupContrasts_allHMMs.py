#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:52:05 2022

@author: okohl


Calculate NABB Metric Group Contrast for all HMM fits:
    1) NABB Metric Group Contrasts are modeled with GLMs accounting for confounds.
    2) Significance of the Group Contrast is assessed with max T-statistic 
       Permutation tests pooling across states. This controls for multiple
       comparisons across states for each state metric separately.
    3) Values are stored for Plotting of Contrast Robustness in next script.
    

"""
import numpy as np
import pickle
import pandas as pd
import glmtools as glm


#%% --- Calculate Burst Metric Group Contrasts ---


# ------------------
# --- load data ----
# ------------------

# ---- Load Confounds ----
behav_in = '/path/to/behav_dir/'
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
number_of_states = [8,10,12] # all states
nrepeate = [1,2,3,4,5,6,7,8,9,10] # all  repetitions

for K in number_of_states: # Loop over different states
    for irun in nrepeate:  # Loop over different HMM inferences

        # Set dirs
        proj_dir = '//path/to/proj_dir/'
        indir = proj_dir + '/Data/Burst_x_State_Metrics/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        outdir_plot = proj_dir + 'Results/State_x_Burst_Metric_GroupContrast/ds' + str(fs) + '/K' + str(K) + '/run' + str(irun) + '/'
        data_outdir = proj_dir + 'Data/Burst_x_State_Metrics_GroupContrast/ds250/K' + str(K) + '/run' + str(irun) + '/'

        # ---- Load Burst Metrics -----
        allMetrics = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metrics']
        MetricLabels = pickle.load(open(indir + 'Burst_x_State_Metrics_GLM_Pow.mat', "rb"))['metric labels']

        # -------------------------------------------------------------------
        # --- GLM comparing NABB Metrics while controlling for confounds ---
        # -------------------------------------------------------------------
             
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
