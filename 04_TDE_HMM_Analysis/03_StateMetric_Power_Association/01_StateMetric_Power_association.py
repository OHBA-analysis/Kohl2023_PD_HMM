
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:00:39 2022

@author: okohl


Script calculates GLMs predicting Power based on sensorimotor State Fractional
Occupancy while controlling for confounds.

The learned model is used in following script to visualise how different levels of FO
influence Power Spectra while holding all other predictors constant.

Saved Data is input for Figure 3.
"""

import glmtools as glm
import numpy as np
import pandas as pd
import scipy.io as io
import pickle

#%% Set Parameters and Dirs

# Parameter
nSub = 67
K = 8
fsample = 250
irun = 7
sub_split = 36  # 36 HCs and 30 PD patients


# Set Dirs
proj_dir = '/path/to/proj_dir/'
in_dir = proj_dir +  'Data/StateMetrics/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

outdir_dat = proj_dir + 'Data/StateMetrics_x_Pow/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'

outdir_plot = proj_dir + 'Results/StateMetrics_x_Pow/ds' + \
    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/allSub/'

pow_dir = proj_dir + '/Data/staticSpectra/'

#%% --- load Covariates ----
behav_in = '/path/to/behavioral_data/'
df = pd.read_csv(behav_in + 'BehaveData_SourceAnalysis.csv')

# Remove Rows with nan and store indices of good Rows
df_in = df[['Group', 'Handedness', 'Gender', 'Age', 'Education']] # UPDRS motor just added to identify Pd patients
in_ind = np.prod(df_in.notna().values, axis=1)
in_ind = np.ma.make_mask(in_ind)
df_in = df_in[in_ind]


#%% --- Load Fractional Occupancies ---

state = 7 # motor state (before state sorting)
labels_metric = 'fractional_occupancy'
labels = ['State1', 'State2', 'State3', 'State4', 'State5', 'State6',
          'State7', 'State8']

in_data = io.loadmat(in_dir + labels_metric + ".mat")['out']
df_states_in = pd.DataFrame(in_data, columns=labels)
df_states = df_states_in[in_ind]


#%% --- Load Power Spectra ---
Level1 = pickle.load(open( pow_dir + "zscore_pow_all.dat", "rb" ) )

# lose low freqs - 1/f can dominate stats
freqs = Level1['Freqs']
keeps = np.where(np.logical_and(freqs>2,freqs <= 30))[0]
freqs = freqs[keeps]

# Grab Copes of interest
copes = Level1['Pow'][in_ind,:,:][:,keeps,:]


#%% --- Loop looping across States --- 
model_all = []
thresh_tmp = []
P_all = []
P_freq_all = []
thresh_freq_tmp = []

for ind, i in enumerate([state]):
    data = glm.data.TrialGLMData(data=copes,
                                 HMM_state=df_states[labels[i]],
                                 covariate=df_in['Age'].values,
                                 gender=df_in['Gender'].values,
                                 handedness=df_in['Handedness'].values,
                                 education=df_in['Education'].values,
                                 num_observations=copes.shape[0])

    DC = glm.design.DesignConfig()
    DC.add_regressor(name='Constant',rtype='Constant')
    DC.add_regressor(name='Gender', rtype='Parametric', datainfo='gender', preproc='z')
    DC.add_regressor(name='Handedness', rtype='Parametric', datainfo='handedness', preproc='z')
    DC.add_regressor(name='Education', rtype='Parametric', datainfo='education', preproc='z')
    DC.add_regressor(name='Age', rtype='Parametric', datainfo='covariate', preproc='z')
    DC.add_regressor(name='HMM_state', rtype='Parametric', datainfo='HMM_state', preproc='z')
    
    # Add Contrast to Model
    DC.add_simple_contrasts()

    # Create design martix
    des = DC.design_from_datainfo(data.info)
    #des.plot_summary(savepath=outdir_plot + 'ModelChecks/State' + str(5) +'/GLM_summary.png')
    #des.plot_efficiency(savepath=outdir_plot + 'ModelChecks/State' + str(5) +'/GLM_efficiency.png')
    
    # fit the model
    model = glm.fit.OLSModel(des,data)
    model_all.append(model)

    
# --- Save Data for Further Plotting elsewhere ---
mdict = {'model_all': model_all, 'freqs':freqs,}
pickle.dump(mdict, open('/home/okohl/Documents/HMM_PD_V07/Data/V2_Plots/FO_Power_Projections.p', "wb"))