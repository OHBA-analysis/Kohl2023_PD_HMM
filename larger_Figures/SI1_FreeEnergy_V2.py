#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:25:22 2022

@author: okohl


Creates Figure Figure SI1

Free Energy values of all 30 HMM fits are loaded and plotted.
HMMs infering the same number of states are plotted on the same line with same color.

For each Line (HMMs with same number of states) the run with the lowest free
Energy is the "best fitting run". This run should be used for further analyses.
In the present Paper we focused on run 1 of the HMMs inferring 8 states.

"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Set Outdors
proj_dir = '/path/to/proj_dir/'

# Load Data
fe8 = loadmat(proj_dir + 'fe_State8.mat')['fe_all'].T
fe10 = loadmat(proj_dir + 'fe_State10.mat')['fe_all'].T
fe12 = loadmat(proj_dir + 'fe_State12.mat')['fe_all'].T

# Get colorMap
cols = plt.cm.Reds(np.linspace(.45, 1, 3))

x1 = range(1,11)
x2 = range(1,11)

#%% Make Line Plots with larger Markers

ax = plt.subplot(111)

# Plot Lines
ax.plot(x1,fe8, lw=2,color = cols[0], linestyle=':', marker='o', markersize=12)
ax.plot(x2,fe10, lw=2, color = cols[1], linestyle=':', marker='o', markersize=12)
ax.plot(x2,fe12,lw=2, color = cols[2], linestyle=':', marker='o', markersize=12)

# Add Scatter
ax.scatter(7,fe8[7-1],40, color = cols[0])
ax.scatter(2,fe10[2-1],40, color = cols[1])
ax.scatter(2,fe12[2-1],40, color = cols[2])

# Add Label to lines
ax.text(x1[-1] + .3 ,fe8[-1] - fe8[-1]*.00005, '8 States', fontsize=14, color = cols[0])
ax.text(x2[-1] + .3 ,fe10[-1] - fe10[-1]*.00005, '10 States', fontsize=14, color = cols[1])
ax.text(x2[-1] + .3 ,fe12[-1] - fe12[-1]*.00005, '12 States', fontsize=14,color = cols[2])

# Axis Labels
ax.set_xlabel('Run',fontsize = 14,labelpad = 15)
ax.set_ylabel('Free Energy',fontsize = 14, labelpad = 15)

# ytick lines
ax.plot(x1, [9.9650e+08] * len(x1), "--", lw=0.5, color="black", alpha=0.3) 
ax.plot(x1, [9.9600e+08] * len(x1), "--", lw=0.5, color="black", alpha=0.3)
ax.plot(x1, [9.9550e+08] * len(x1), "--", lw=0.5, color="black", alpha=0.3) 
ax.plot(x1, [9.9500e+08] * len(x1), "--", lw=0.5, color="black", alpha=0.3)     
ax.plot(x1, [9.9450e+08] * len(x1), "--", lw=0.5, color="black", alpha=0.3) 


ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off


# Adjust Spines
ax.spines["top"].set_visible(False)  
ax.spines["bottom"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.spines["left"].set_visible(False)

