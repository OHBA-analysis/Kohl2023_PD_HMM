#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:20:48 2022

@author: okohl

Script asessing association between State Metrics and Bradykinesia/Rigidity scores.
State Metrics are correlated with UPDRS scores while controlling for confounds
with GLM approach.

Significance of correlations assessed with Max-Tstatistic Permutation tests
pooling across States.

This preocedure is repeated for each State Metric.

Adjusting runs variable to vector with different run numbers allows to run
analysis across all runs with the same number of states. This might be usefull
when assessing robustness of associations.

Importantly: this script has to be run separately for each number of states
--> Fix this

"""

import glmtools as glm
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle


# Def Function to get Labels of states of interest across runs
def get_StateLabels(State_OI):  
        labels = ['State' + str(state_OI[0]), 'State' + str(state_OI[1]), 'State' + str(state_OI[2]), 
                      'State' + str(state_OI[3]), 'State' + str(state_OI[4]), 'State' + str(state_OI[5]),
                      'State' + str(state_OI[6]), 'State' + str(state_OI[7]), 'State' + str(state_OI[8]),
                      'State' + str(state_OI[9])]
        return labels


#%% --- Define across what to loop ---

# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [1,2,3,4,5,6,7,8,9,10]
sampling_rates = [250]
number_of_states = [12]
embeddings = [7]
sig_state_8 = 7 # State with significant effect in HMM with lowest free energy // Sig State -1 because of Python indexing


# Translate sig state in 8K HMM to states in other K HMMs
if number_of_states[0] == 8:
    sig_state = sig_state_8
elif number_of_states[0] > 8:
    # Load lowest FE HMM-Model state matching
    StateMatching_file = '/path/to/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(number_of_states[0]) + '_to_8K_RefHMM_matching.mat'
    sig_state =  loadmat(StateMatching_file)['assig'][:,sig_state_8-1][0]


# --- Load state assignments from State matching ---
StateMatching_file = '/path/to/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(number_of_states[0]) + '/StateMatching.mat'
state_OI = loadmat(StateMatching_file)['assig_all'][:,sig_state-1]


# --- Load UPDRS scores ---
behav_in = '/path/to/behavioral_data/'
behav = pd.read_pickle(behav_in + 'sourceBehave_PDfactors.pkl')

# Remove Rows with nan and store indices of good Rows
df_PD = behav[['Age','Education','UPDRS-Motor',
            'Midline Function', 'Rest Tremor', 'Rigidity','Bradykinesia Right Upper Extremity',
            'Bradykinesia Left Upper Extremity','Postural & Kinetic Tremor',
            'Lower Limb Bradykinesia','Years Since Diagnosis', 'Lateralisation Score']]
in_ind_PD = np.prod(df_PD.notna().values,axis = 1)
in_ind_PD = np.ma.make_mask(in_ind_PD)
df_in_PD = df_PD[in_ind_PD]

df_in_PD['Bradykinesia/Rigidity'] = df_in_PD['Bradykinesia Right Upper Extremity'] + df_in_PD['Bradykinesia Left Upper Extremity'] + df_in_PD['Lower Limb Bradykinesia'] + df_in_PD['Rigidity']
df_in_PD['Tremor'] = df_in_PD['Rest Tremor'] + df_in_PD['Postural & Kinetic Tremor']

labels_PD = ['Bradykinesia/Rigidity','Tremor']


# --- load behavioral data put the in dictionary of covariates ---
behav_in = '/ohba/pi/knobre/PD_data/TDE_HMM_project/data/behavioral_data/'
df = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')


# Remove Rows with nan and store indices of good Rows
df_in = df[['Group','Handedness','Gender','Age','Education']]
in_ind = np.prod(df_in.notna().values,axis = 1)
in_ind = np.ma.make_mask(in_ind)
df_in = df_in[in_ind_PD]


# --- Set Labels for Loop ---
labels_metric = ['fractional_occupancy', 'mean_lifetimes', 
                 'state_rates', 'mean_intervaltimes','State_GLM_BetaPower']

labels = ['State1', 'State2', 'State3', 'State4', 
          'State5', 'State6','State7', 'State8',
          'State9','State10','State11','State12']

OI_labels = get_StateLabels(state_OI)

#%% Start Loop Calculating State Metric x UPDRS associaions for different HMMs

for irun in runs:
    for fsample in sampling_rates:
        for emb in embeddings:
            for K in number_of_states:
                
                print('Running Analysis for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
      
                # ----------------------------------------------------
                #%% GLMs Calculating HMM State Metric UPDRS score associations
                # -----------------------------------------------------
                # Script uses same GLM-Models as static PowSpec Group Comparison
                # 1) GLMs are calculated
                # 2) Permutation Tests assessing significance of 
                #    predictors are calculated
                # ----------------------------------------------------
                
                # Set Outdors
                proj_dir = '/path/to/proj_dir/'
                indir = proj_dir +  'Data/StateMetrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                outdir_dat = proj_dir + 'Data/StateMetrics_x_UPDRS/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                outdir_plot = proj_dir + 'Results/StateMetrics_x_UPDRS/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

                # Get State Labels
                labels_OI = OI_labels[irun-1]


                # --- Loop across State Metrics to calculate GLMs ---
                for iLabel in range(len(labels_metric)):
                    
                    # --- Load State Metrics ---
                    in_data = loadmat(indir + labels_metric[iLabel] + ".mat")['out']
                    df_states_in = pd.DataFrame(in_data, columns=labels[:K])
                    df_states_in = df_states_in[labels_OI]
                    df_states = df_states_in[in_ind_PD]
                    
                    # --- Pre Allocate Vars ---
                    model_all = []
                    P_all = []
                    P_freq_all = []
                    thresh_tmp = []
                    thresh_freq_tmp = []
                    tstats = np.empty([1 ,7, len(labels_OI)])
                    tstats_freq = np.empty([1 ,7, len(labels_OI)])
                    mask = np.empty([1, 7, len(labels_OI)])
                    mask_freq = np.empty([1, 7,len(labels_OI)])
                    p_all_cor = []
                    sig_cor = []
                    tstats_all = []
                                
                    # Define Data for GLM
                    data = glm.data.TrialGLMData(data=df_states.values[:],
                                                 Brad_Rigidity=df_in_PD[labels_PD[0]],
                                                 Tremor=df_in_PD[labels_PD[1]],
                                                 covariate=df_in['Age'].values,
                                                 gender=df_in['Gender'].values,
                                                 handedness=df_in['Handedness'].values,
                                                 education=df_in['Education'].values,
                                                 num_observations=df_in_PD[labels_PD].shape[0])
                
                    # Define Model for GLM
                    DC = glm.design.DesignConfig()
                    DC.add_regressor(name='Constant',rtype='Constant')
                    DC.add_regressor(name='Gender', rtype='Parametric',
                                     datainfo='gender', preproc='z')
                    DC.add_regressor(name='Handedness', rtype='Parametric',
                                     datainfo='handedness', preproc='z')
                    DC.add_regressor(name='Education', rtype='Parametric',
                                     datainfo='education', preproc='z')
                    DC.add_regressor(name='Age', rtype='Parametric',
                                     datainfo='covariate', preproc='z')
                    DC.add_regressor(name='Brad_Rigidity', rtype='Parametric',
                                     datainfo='Brad_Rigidity', preproc='z')
                    DC.add_regressor(name='Tremor', rtype='Parametric',
                             datainfo='Tremor', preproc='z')
                    
                    # Add Contrast   
                    DC.add_simple_contrasts()
                
                    # Create design martix
                    des = DC.design_from_datainfo(data.info)
                    #des.plot_summary(savepath=outdir_plot + 'Model_Checks/State' + str(K) +'_GLM_summary.png')
                    #des.plot_efficiency(savepath=outdir_plot + 'Model_Checks/State' + str(K) +'_GLM_efficiency.png')

                    # --- fit the model ---
                    model = glm.fit.OLSModel(des,data)
                    
                    
                    #%% --- Test for significance with Permutation tests ---
                    for iContrast in range(5,7):

                        # --- Permutation tests pooling across States ---
                        
                        contrast = iContrast # select the UPDRS contrast
                        metric = 'tstats' # add the t-stats to the null rather than the copes
                        nperms = 10000  # for a real analysis 1000+ is best but 100 is a reasonable approximation.
                        nprocesses = 4 # number of parallel processing cores to use
                        pooled_dims = ()
                        
                        
                        P = glm.permutations.MaxStatPermutation(des, data, contrast, nperms,
                                                                metric=metric, nprocesses=nprocesses,
                                                                pooled_dims = pooled_dims)
                        
                        P_all.append(P)

                    
                        # Bring outout into [state metric][State] shape
                        thresh = P.get_thresh([95, 99]) 
                        
                        p_all_cor.append(thresh)                                   
                        sig_cor.append(abs(model.tstats[iContrast]) > thresh[0])
                        tstats_all.append(model.tstats[iContrast])
                        
                        
                    #%% --- Save Outputs ---
                    mdict = {'tstats': tstats_all, 'p_cor': p_all_cor, 'Sig_cor':sig_cor,
                             'model': model, 'Perm_cor': P_all, 'Metrics': labels_metric, 'Contrasts': labels_PD}
                    pickle.dump( mdict, open( outdir_dat + labels_metric[iLabel] +"_x_UPDRS_GLM_State" + str(state_OI[irun-1]) + ".mat", "wb" ) )
                    
                    pickle.load( open( outdir_dat + labels_metric[iLabel] +"_x_UPDRS_GLM_State" + str(state_OI[irun-1]) + ".mat", "rb" ) )
                    