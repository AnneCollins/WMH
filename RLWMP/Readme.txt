% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

DATASETS
There is one dataset in the DataSets folder, named Expe14,mat, which corresponds to the RLWMP data set in the paper (McD). 

MAIN SCRIPTS TO RUN:
[make sure to CD into the folder before running scripts]
- validation.m: will plot behavior for the datasets and selected models, is used to generate Fig. 3B. Calls the function “analyzeBehavior.m”; requires fit parameters in the /Fits folder, and calls RLWM.m in the /Models folder. Selected models can be customized in the script.
- ModelComparison.m: will plot model comparison for all datasets and selected models. Necessitates the spm12 package for the spm_mbs function, included in the /spm folder. This produces figure 3A. Also plots model parameters from figure S12.

SECONDARY SCRIPTS TO RUN:
- general_fitting_script_RL2.m runs the model fitting for all models. Models  to run can be customized. This is a slow script, but it is set in fast demo mode. Outputs are saved in /Fits. New outputs will be saved in /NewFits folder. 
- Confusion_Gen_Rec.m plots model and parameter identification for figures S13 and S14 and S13. If regenerateGenRec is set to true, it runs the model and parameter identification script for selected models (customizable) and saves it in /GenRec - this is SLOW, except in demo mode (Demo=true). If false, it only plots

OTHER functions:
- analyzeBehavior_RLWMP.m analyzes a behavior data set and outputs summary statistics, and is called by plotBehavior.m and validation.m
- test consistency.m tests model definition consistency, and is called by general_fitting_scripts_RL2.m.
- Models/RL2.m simulated model behavior for multiple models, specified in input, and outputs data in specified format.
- Models/RL2_llh.m computes the likelihood of data for multiple models, specified in input.
 