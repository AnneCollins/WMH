% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

DATASETS
There are 6 datasets in the DataSets folder, named ExpeX,mat, with X=1,2,3,5,12,16. They correspond to 'CF12','SZ','EEG','fMRI','Dev','GL' in the paper. 

MAIN SCRIPTS TO RUN:
[make sure to CD into the folder before running scripts]
- PlotBehavior.m: will plot behavior for all datasets, is used to generate Fig. 1. Calls the function “analyzeBehavior.m”
- validation.m: will plot behavior for all datasets and selected models, is used to generate Fig. 2B,C, and supplementary figures. Calls the function “analyzeBehavior.m”; requires fit parameters in the /Fits folder, and calls RLWM.m in the /Models folder. Selected models can be customized in the script.
- ModelComparison.m: will plot model comparison for all datasets and selected models. Necessitates the spm12 package for the spm_mbs function, included in the /spm folder. This produces figure 2A, as well as supplementary figures s6 and S7

SECONDARY SCRIPTS TO RUN:
- general_fitting_script.m runs the model fitting for all models and data sets. Models and datasets to run can be customized. This is a slow script (multiple days for all models and datasets). It is currently set in demo mode to run fast (fewer models and participants). Full outputs are saved in /Fits. Any new outputs will be saved in /NewFits folder. 
- Confusion_Gen_Rec.m plots model and parameter identification for figures S8 and S9. If runGenRec is set to true, it runs the model and parameter identification script for selected models and datasets (customizable) and saves it in /GenRec - this is SLOW (a few hours to a day depending on dataset, hardware and parallelization).  

OTHER functions:
- analyzeBehavior.m analyzes a behavior data set and outputs summary statistics, and is called by plotBehavior.m and validation.m
- Models/RLWM.m simulated model behavior for multiple models, specified in input, and outputs data in specified format.
- Models/RLWM_llh.m computes the likelihood of data for multiple models, specified in input.
 