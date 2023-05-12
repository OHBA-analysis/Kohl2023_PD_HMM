#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:38:36 2022

@author: okohl


Script to plot association between specific state metric and Bradykinesia/Ridgity score.
This was only used to check whether state metrics differing significantly between
the two groups, also demonstrated significant associations with symptom severity
scores.

"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
StateMatching_file = '/home/okohl/Documents/HMM_PD_V07/Data/StateMatching/ds' + str(sampling_rates[0]) + '/K' + str(number_of_states[0]) + '/StateMatching.mat'
state_OI = loadmat(StateMatching_file)['assig_all'][:,sig_state-1]



for irun in runs:
    for fsample in sampling_rates:
        for emb in embeddings:
            for K in number_of_states:
                
                print('Plott Correlation for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
                                     
                
                # --- Set Dirs
                indir_metrics = '/home/okohl/Documents/HMM_PD_V07/Data/StateMetrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                indir_behav = '/ohba/pi/knobre/PD_data/TDE_HMM_project/data/behavioral_data/'
                
                indir_GLM = '/home/okohl/Documents/HMM_PD_V07/Data/StateMetrics_x_UPDRS/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                outdir_plot = '/home/okohl/Documents/HMM_PD_V07/Results/StateMetrics_x_UPDRS/ds' + \
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
                labels_metric = ['fractional_occupancy', 'mean_lifetimes', 'state_rates', 'mean_intervaltimes', 'State_GLM_BetaPower']
                labels_metric_plot = ['Fractional Occupancy', 'Mean Lifetimes (sec)', 'State Rates', 'Mean Interval Times (sec)', 'Beta Power (a.u.)']
                labels_all = ['State1', 'State2', 'State3', 'State4', 'State5', 'State6',
                           'State7', 'State8','State9','State10']
                
                OI_labels = get_StateLabels(number_of_states[0],state_OI)

                OI_num = state_OI
                
                
                # ---------------------------------
                #%% Load GLM outpus
                iLabel = 4 # To Load Interval Times
                model = pickle.load( open( indir_GLM + labels_metric[iLabel] +"_x_UPDRS_GLM_State"+ str(sig_state) +".mat", "rb" ) )['model']
                beta = model.betas[5]
                tstat = model.tstats[5]
                
                nulls = pickle.load( open( indir_GLM + labels_metric[iLabel] +"_x_UPDRS_GLM_State"+ str(sig_state) +".mat", "rb" ) )['Perm_cor'][1].nulls
                
                # Get p - two sided/undirected
                p = (abs(tstat)<abs(nulls)).sum()/nulls.size
                
                # --------------------------------
                #%% Load State Metric Data
                in_data = loadmat(indir_metrics + labels_metric[iLabel] + ".mat")['out']
                df_states_in = pd.DataFrame(in_data, columns=labels_all[:K])
                df_states_in = df_states_in[OI_labels[irun-1]]
                df_states = df_states_in[in_ind_PD]
                 
                #%% Start Plotting

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
                sns.regplot(x="Bradykinesia/Rigidity", y="Interval Times", data=dat_plot, ax = ax, color = '#3B5E8C' )
                
                # Axis Labels
                ax.set_ylabel('State ' + str(OI_num[irun-1]) + '\n' + labels_metric_plot[iLabel], fontsize = 12, labelpad = 10)
                ax.set_xlabel('Bradykinesia/\nRigidity', fontsize = 12, labelpad = 5)
                
                # Adjust Ticks
                ax.tick_params(axis='x', labelsize= 12)
                ax.tick_params(axis='y', labelsize= 12)
                
                # Add Text about Statistics
                ax.text(.95,.98,'T-Stat = ' + str(np.round(tstat,2)[0]) + ', p = ' + str(np.round(p,2)),
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform = ax.transAxes,
                        fontsize = 14)
                
                # Remove Box Around Subplot
                sns.despine(ax=ax, top=True, right=True, left=False,
                        bottom=False, offset=None, trim=False)
                
                # ----- Save Figure ------
                plt.savefig(outdir_plot + labels_metric[iLabel] + '_UPDRSfact_GLM_corr_State' + str(sig_state) +'_overview.svg',
                               transparent=True, bbox_inches="tight",format='svg')