# Sensorimotor-Network-Dynamics-in-PD
Scripts for "Sensorimotor network dynamics changed in resting-state MEG recordings of Parkinson’s Disease Patients" manuscript.


## Requirements

Preprocessing and TDE-HMMs were run in MATLAB 2020a using the following Toolboxes:  
    - [OSL](https://github.com/OHBA-analysis/osl-docs)        
    - [HMM-Mar Toolbox](https://github.com/OHBA-analysis/HMM-MAR)
    
Calulations of State-Specific Spectra, Burst Analysis, and Statistics calculated in Python:  
    - all scripts can be run in the combined osl & osl-dynamics environment.  
      Instructions for the environment set up can be found [here](https://github.com/OHBA-analysis/osl-dynamics#installing-within-an-osl-environment).


## Contents

```01_preprocessing```: Script doing HMM-specififc preprocessing prior to running the HMMs.  
```02_Static_Power_Spectrum```: Scripts calculating static power spectra and calculating group comparisons.  
```03_Burst_Analysis```: Scripts extracting burst metrics and contrasting them between groups.  
```04_TDE_HMM_Analysis```: Scripts calculating TDE-HMMs and State Metrics from preprocessed data.  
```05_Overlap_Analysis```: Scripts investigating overlap between whole-brain networks inferred with the TDE-HMM and beta bursts.  
```06_NABB_Analysis```: Scripts calculating metrics of Network Associated Beta Bursts (NABBs) and comparing them between groups.  
```helpers```: Scripts with helper functions for plotting, burst analysis, and overlap analysis.    
```larger_Figures```: Scripts to generate large overview Figures for manuscript.
