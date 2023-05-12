#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 10:52:30 2022

@author: okohl

Script Creats Figure SI2

State Metrics of the HMM-States are loaded and values of all participants are
plotted in violine plots.

Especially, Fractional Occupancy plot is important to check to see whether HMM
states are mixing well. Values close to 1 indicate that single particpants are
represented by single states - this indicates a bad model fit.

"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import seaborn as sns

# --- Set Parameters Specifying for which HMM tests are calculated -------
runs = [7]
sampling_rates = [250]
number_of_states = [8]
embeddings = [7]
for irun in runs:
    for fsample in sampling_rates:
        for emb in embeddings:
            for K in number_of_states:
                
                print('Running Analysis for run' + str(irun) + 'of ' + str(K) + 'K_emb' + str(emb) + ' with Sampling Rate of ' + str(fsample) + ' run' + str(irun))
      
                
                outdir_plot = '/path/to/Results/StateMetric_GroupContrast/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
 
                
                # ---------------------------------------------------
                # %% Make Plots for State Metric Group Comparison GLMs
                # ---------------------------------------------------
                # Script creates split Violin Plots to depict group 
                # differences in State Metrics.
                #
                # Significant Differences are marked with ** or *
                # ----------------------------------------------------

                
                #%% Load Custom Made Functions
                # ---------------------------
                def p_for_annotation(p):
                    if p >= .05:
                        p_plt = .1
                    elif (.05 > p >= .01):
                        p_plt = .01
                    elif (.01 > p >= .001):
                        p_plt = .001
                    else:
                        p_plt = 'p < .001'
                    return p_plt
                
                
                def split_barplot_annotate_brackets(num1, num2, data, center, height, ax_in=plt, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
                    """ 
                    Annotate barplot with p-values.
                
                    :param num1: number of left bar to put bracket over
                    :param num2: number of right bar to put bracket over
                    :param data: string to write or number for generating asterixes
                    :param center: centers of all bars (like plt.bar() input)
                    :param height: heights of all bars (like plt.bar() input)
                    :param yerr: yerrs of all bars (like plt.bar() input)
                    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
                    :param barh: bar height in axes coordinates (0 to 1)
                    :param fs: font size
                    :param maxasterix: maximum number of asterixes to write (for very small p-values)
                    """
                
                    if type(data) is str:
                        text = data
                    else:
                        # * is p < 0.05
                        # ** is p < 0.005
                        # *** is p < 0.0005
                        # etc.
                        text = ''
                        p = .05
                
                        while data < p:
                            text += '*'
                            p /= 10.
                
                            if maxasterix and len(text) == maxasterix:
                                break
                
                        if len(text) == 0:
                            text = ''
                
                    lx, ly = center[num1], height[num1]
                    rx, ry = center[num2], height[num2]
                
                    if yerr:
                        ly += yerr[num1]
                        ry += yerr[num2]
                
                    ax_y0, ax_y1 = plt.gca().get_ylim()
                    dh *= (ax_y1 - ax_y0)
                    barh *= (ax_y1 - ax_y0)
                
                    y = max(ly, ry) + dh
                
                    barx = [lx, lx, rx, rx]
                    bary = [y, y+barh, y+barh, y]
                    mid = ((lx+rx)/2, y+barh)
                    mid = ((lx+rx)/2, y+barh)
                
                    #ax_in.plot(barx, bary, c='black')
                
                    kwargs = dict(ha='center', va='bottom')
                    if fs is not None:
                        kwargs['fontsize'] = fs
                
                    ax_in.text(*mid, text, **kwargs)
                    
                # ------------------------------             
                
                HMM_indir = '/path/to/Data/StateMetrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                
                GLM_indir = '/path/to/Data/StateMetrics/ds' + \
                    str(fsample) + '/K' + str(K) + '/run' + str(irun) + '/'
                    
                                
                #%% Get Subjects that are in Source space analysis
                subIDs = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19',
                                 '51','52','53','54','56','57','58','59','61','62','63','64','65','66','67','68','69','70','71',
                                 '101','102','103','104','105','106','107','109','110','116','117','118',
                                 '151','153','154','155','156','157','159','160','161','162','163','164','165','166','167','168','169','170']
                
                HCsub = np.array(range(0, subIDs.index('71')+1))
                PDsub = np.array(range(HCsub[-1]+1, subIDs.index('170')+1))
                allSub = np.hstack([HCsub, PDsub])
                
                group_vect = np.ones_like(allSub)
                group_vect[allSub > len(HCsub)-1] = 2
                
                
                #%% Define Labels
                # To Load State Metrics
                labels_metric = ['fractional_occupancy','mean_lifetimes', 'mean_intervaltimes']
                
                
                # For Plotting
                labels = ['State \n 1', 'State \n 2', 'State \n 3', 'State \n 4', 'State \n 5', 'State \n 6',
                          'State \n 7', 'State \n 8', 'State \n 9', 'State \n 10', 'State \n 11', 'State \n 12']
                labels_plot = ['Fractional Occupancy', 'Lifetimes (sec)',
                               'Intervaltimes (sec)']
                
                # Load State Sorting vector
                sort_states = np.load('/path/to/sortingVector/ClusterPowSorting.npy')
                
                #%% Set up Grid for Plotting
                
                x = 1.2
                fig = plt.figure(dpi=300, figsize=(12.8*x, 4*x))
                gs = plt.GridSpec(1, 3, height_ratios=[1], wspace=.4, hspace=.1)
                ax = np.zeros(4, dtype=object)
                ax[0] = fig.add_subplot(gs[0, 0])
                ax[1] = fig.add_subplot(gs[0, 1])
                ax[2] = fig.add_subplot(gs[0, 2])
                
                
                for iMetric, metric in enumerate(labels_metric): # -1 to exclude switch rates
                    
                    # Load importantb vars from GLM
                    GLM_in = pickle.load(open( GLM_indir + metric + "_GroupComp_GLM.mat", "rb" ) )
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
                   
                    
                    #%%  Violine Plot                  
                    # Colors
                    col = "#dedede"

                    # Make Violine Plots
                    sns.violinplot(x="State", y=metric, inner = "quartile",
                                        data=df_fact, color=col, 
                                        cut=.5 , bw=.5 ,ax = ax[iMetric])
                    
                    for ind, violin in enumerate(ax[iMetric].findobj(PolyCollection)):
                        violin.set_edgecolor('k')
                        violin.set_linewidth(.8)
                    
                    # Set Line Colors
                    for l in ax[iMetric].lines:
                        l.set_linestyle('--')
                        l.set_linewidth(.8)
                        l.set_color('k')
                        l.set_alpha(0.8)
                        
                    for l in ax[iMetric].lines[1::3]:
                        l.set_linestyle('-')
                        l.set_linewidth(1)
                        l.set_color('k')
                        l.set_alpha(.8)
                    
                    # Labels and Ticks
                    ax[iMetric].set_xlabel('State', fontsize=16)
                        
                    ax[iMetric].set_xticklabels(np.arange(1,K+1), fontsize=16) #becasue of np.arrang does not include end of specified range
                    ax[iMetric].set_ylabel(labels_plot[iMetric])
                    ax[iMetric].yaxis.label.set_size(16)
                    ax[iMetric].tick_params(axis='both', which='major', labelsize=14)
                    ax[iMetric].tick_params(axis='both', which='minor', labelsize=14)
                    
                    # Legend
                    if iMetric == 1:
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
                    
                    
plt.savefig('/path/to/SI_StateMetrics_Overall.svg',transparent = True,bbox_inches="tight", 
            format='svg')
                   