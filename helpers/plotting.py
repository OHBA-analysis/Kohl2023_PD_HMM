#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:00:38 2022

@author: okohl


Create Color Dictionary

"""

from matplotlib.colors import to_rgb
import numpy as np
import matplotlib.pyplot as plt

def get_colors():
    
    # ---- PD vs HC -----
    
    # Bin PD vs HC colors 
    PDvsHC_bin = ['#646C73', '#2477BF']

    # PD vs HC color pallete vor violine plots
    # Last 8 violines are shaded (for states with power smaller than average)
    grey = to_rgb(PDvsHC_bin[0])
    blue = to_rgb(PDvsHC_bin[1])
    wgrey = .6 + .5 * np.array(grey)
    wblue = .6 + .5 * np.array(blue)

    PDvsHC_palette = [grey, blue, grey, blue, grey, blue, grey, blue,
                      wgrey, wblue, wgrey, wblue, wgrey, wblue, wgrey, wblue]
    
    
    # ---- Empirical vs Null ----
    EmpvsNull_bin = ['#FDD2CC','#BF3C37']
    
    # PD vs HC color pallete vor violine plots
    # Last 8 violines are shaded (for states with power smaller than average)
    grey = to_rgb(EmpvsNull_bin[0])
    red = to_rgb(EmpvsNull_bin[1])
    wgrey = .5 + .5 * np.array(grey)
    wred = .5 + .5 * np.array(red)

    EmpvsNull_palette = [grey, red, grey, red, grey, red, grey, red,
                      wgrey, wred, wgrey, wred, wgrey, wred, wgrey, wred]
    
    cols = {'PDvsHC_bin': PDvsHC_bin, 'PDvsHC_palette': PDvsHC_palette,
            'EmpvsNull_bin' : EmpvsNull_bin, 'EmpvsNull_palette': EmpvsNull_palette}
    
    return cols



# --- Plotting Functions ---

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


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


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

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax_in.text(*mid, text, **kwargs)
    
    
    
    
