#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:38:36 2022

@author: okohl

Load GLM and Max-Tstatistic Permutation Test outputs.
Load State Metric with significant Group Difference and Bradykinsia/Rigidity scores.
Plot scatter plot of state metric and Bradykinesia/Rigidity scores.
Add tstats and p-values from GLM/Max tstatistic permutation tests.
"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
#import ptitprince as pt

import sys
sys.path.append("/path/to/helpers/")
from plotting import get_colors

sns.set_style("ticks")


# Def Function to get Labels of states of interest across runs
def get_StateLabels(K,State_OI):
        labels = ['State' + str(state_OI[0]), 'State' + str(state_OI[1]), 'State' + str(state_OI[2]), 
                    'State' + str(state_OI[3]), 'State' + str(state_OI[4]), 'State' + str(state_OI[5]),
                    'State' + str(state_OI[6]), 'State' + str(state_OI[7]), 'State' + str(state_OI[8]),
                    'State' + str(state_OI[9])]
        return labels


#%% --- Define across what to loop ---

# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [7]
sampling_rates = [250]
number_of_states = [8]
embeddings = [7]
sig_state = 8 # State with significant effect in HMM with lowest free energy // Sig State -1 because of Python indexing


# --- Load state assignments from State matching ---
StateMatching_file = '/path/to/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(number_of_states[0]) + '/StateMatching.mat'
state_OI = loadmat(StateMatching_file)['assig_all'][:,sig_state-1]



for irun in runs:
    for fsample in sampling_rates:
        for emb in embeddings:
            for K in number_of_states:
                
                print('Plott Correlation for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
                                     
                
                # --- Set Dirs
                indir_metrics = '/path/to/Data/Burst_x_State_Metrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                indir_behav = '/path/to/behavioral_data/'
                
                indir_GLM = '/path/to/Data/Burst_x_State_Metrics_x_UPDRS/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                outdir_plot = '/path/to/Results/State_x_Burst_Metric_x_UPDRS/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/State_OI/'
                
                
                #%% Load UPDRS scores 
                behav = pd.read_pickle(indir_behav + 'sourceBehave_PDfactors.pkl')
                
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
                
                
                labels_PD = ['Bradykinesia/\nRigidity','Tremor']
                
                # ---------------------------------
                #%% Load fractional occupancies
                labels_metric = ['fractional_occupancy', 'mean_lifetimes', 'mean_intervaltimes','state_rates', 'beta_pow']
                labels_metric_plot = ['Fractional Occupancy', 'Mean Life Times', 'Mean Interval Times', 'Burst Rates', 'Beta Power']
                labels_all = ['State1', 'State2', 'State3', 'State4', 'State5', 'State6',
                           'State7', 'State8','State9','State10']
                
                OI_labels = get_StateLabels(number_of_states[0],state_OI)
                OI_num = state_OI
                                
                # ---------------------------------
                #%% Load GLM outpus
                iLabel = 0 # To Load Fractional Occupancy
                model = pickle.load( open( indir_GLM + labels_metric[iLabel] +"_x_UPDRS_GLM_State"+ str(sig_state) +".mat", "rb" ) )['model']
                beta = model.betas[5]
                tstat = model.tstats[5]
                
                nulls = pickle.load( open( indir_GLM + labels_metric[iLabel] +"_x_UPDRS_GLM_State"+ str(sig_state) +".mat", "rb" ) )['Perm_cor'][1].nulls
                
                # Get p - two sided/undirected
                p = (abs(tstat)<abs(nulls)).sum()/nulls.size
                
                # --------------------------------
                #%% Load State Metric Data
                in_data = pickle.load(open(indir_metrics + "Burst_x_State_Metrics_GLM_Pow.mat","rb"))['metrics'][iLabel]
                in_data = np.squeeze(in_data)
                df_states_in = pd.DataFrame(in_data, columns=labels_all[:K])
                df_states_in = df_states_in[OI_labels[irun-1]]
                df_states = df_states_in[in_ind_PD]
                 
                #%% Start Plotting

                # --- Get colors ---
                cols = get_colors()['PDvsHC_bin']

                # ---- Set Up ------
                fig = plt.figure(dpi=300)
                gs = plt.GridSpec(1, 1, wspace=.6, hspace=.32)
                ax = np.zeros(1, dtype=object)
                ax = fig.add_subplot()
                
                # ------  Do Regression Plots ----------
                dat_plot = {'Interval Times': df_states.values,
                            'Bradykinesia/Rigidity' : df_in_PD['Bradykinesia/Rigidity']}
                dat_plot = pd.DataFrame(dat_plot)
                
                sns.set_style("ticks")
                sns.regplot(x="Bradykinesia/Rigidity", y="Interval Times", data=dat_plot, ax = ax, color = cols[1])
             
                # Labels and Tickx
                ax.set_ylabel('Frontal NAAB\n' + labels_metric_plot[iLabel], fontsize = 12, labelpad = 10)
                ax.set_xlabel('Bradykinesia/\nRigidity', fontsize = 12, labelpad = 5)
                
                ax.tick_params(axis='x', labelsize= 12)
                ax.tick_params(axis='y', labelsize= 12)
                
                # Add Text adding Stats 
                ax.text(.95,.98,'T-Stat = ' + str(np.round(tstat,2)[0]) + ', p = ' + str(np.round(p,2)),
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform = ax.transAxes,
                        fontsize = 14)
                
                # Remove Box Around Subplot
                sns.despine(ax=ax, top=True, right=True, left=False,
                        bottom=False, offset=None, trim=False)
                
                # ----- Save Figure ------
                plt.savefig(outdir_plot + labels_metric[iLabel] + '_BurstState_UPDRSfact_GLM_corr_State' + str(sig_state) +'_overview.svg',
                               transparent=True, bbox_inches="tight",format='svg')
                
                
                # ------ Results for further Plotting ------
                mdict = {'Data': dat_plot, 'tstat': tstat, 'p': p}
                pickle.dump(mdict, open( "/path/to/Data/V2_Plots/" 
                                        + labels_metric[iLabel] +
                                        "NAAB_correl_Data.mat", "wb"))
